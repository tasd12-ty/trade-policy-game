"""LLM 博弈实验（两国同回合同步决策，由 LLM 生成策略）。

与 grad_game.py 平行，使用 LLM 替代梯度优化生成策略。

运行：
    1) 在文件底部 `if __name__ == "__main__":` 中编辑 cfg（包括 LLM 方/计算方、模型、超参等）；
    2) 设置对应平台的 API Key 环境变量（如 DeepSeek: `DEEPSEEK_API_KEY`）；
    3) 直接运行：python analysis/optimization/llm_game.py
"""

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model import (
    bootstrap_simulator,
    TwoCountryDynamicSimulator,
    create_symmetric_parameters,
    CountryState,
    DEFAULT_DEVICE,
    TORCH_DTYPE,
)
from analysis.optimization.plotting import plot_game_analysis, plot_supply_demand_gap
from analysis.optimization.grad_game import (
    TriggerConfig,
    ConstraintsConfig,
    ObjectiveConfig,
    OptimizationConfig,
    DifferentiableObjective,
    _apply_trigger,
    _policy_dict_from_vector,
    _current_tariff_rate,
    _current_quota_multiplier,
    _history_slice,
    _apply_policy_differentiable,
    _make_optimizer,
)
from analysis.optimization.llm import LLMPolicyAgent, OpenAIClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Country = Literal["H", "F"]


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置。"""
    model: str = "qwen-plus"
    preset: Literal["qwen", "openai", "deepseek"] = "qwen"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # 设为 None 表示不显式传 temperature（由服务端/模型默认策略决定）
    temperature: Optional[float] = None
    # 设为 None 表示不显式传 max_tokens（由服务端/模型默认策略决定；仍受模型上限约束）
    max_tokens: Optional[int] = 2048


@dataclass(frozen=True)
class LLMGameConfig:
    """LLM 博弈实验配置。"""
    name: str = "llm_game"
    theta_price: float = 0.1
    normalize_gap_by_supply: bool = True
    rounds: int = 10
    decision_interval: int = 10
    lookahead_periods: int = 10  # 用于梯度优化的前瞻期数
    warmup_periods: int = 1000
    trigger_settle_periods: int = 0
    trigger: Optional[TriggerConfig] = None
    constraints: ConstraintsConfig = ConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    llm: LLMConfig = LLMConfig()
    # LLM 控制哪方: "H", "F", 或 "both"
    llm_plays: Literal["H", "F", "both"] = "both"
    # 混合模式：非 LLM 一方使用什么策略
    # - "fixed": 保持当前策略不变
    # - "gradient": 使用梯度优化
    non_llm_strategy: Literal["fixed", "gradient"] = "fixed"
    # 梯度优化配置（仅 non_llm_strategy="gradient" 时使用）
    opt_config: OptimizationConfig = OptimizationConfig()
    plot: bool = True
    output_dir: str = "results_llm"


def _optimize_single_country_gradient(
    sim: TwoCountryDynamicSimulator,
    country: Country,
    opponent_policy: Dict[str, Dict[int, float]],
    *,
    config: LLMGameConfig,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """为单个国家运行梯度优化，对手策略固定。
    
    Args:
        sim: 仿真器
        country: 要优化的国家
        opponent_policy: 对手的固定策略 {"tariff": {...}, "quota": {...}}
        config: 博弈配置
        
    Returns:
        (policies, info) where policies = {"tariff": {...}, "quota": {...}}
    """
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    opponent: Country = "F" if country == "H" else "H"
    
    # 构建约束掩码
    active_mask = torch.zeros(n_sectors, dtype=torch.bool, device=DEFAULT_DEVICE)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True
    
    # 获取当前策略
    init_tau = _current_tariff_rate(sim, country).to(DEFAULT_DEVICE)
    init_quota = _current_quota_multiplier(sim, country).to(DEFAULT_DEVICE)
    
    # 对手策略转换为 tensor
    opp_tau_dict = opponent_policy["tariff"]
    opp_quota_dict = opponent_policy["quota"]
    opp_tau = torch.zeros(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
    opp_quota = torch.ones(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
    for s, v in opp_tau_dict.items():
        opp_tau[int(s)] = float(v)
    for s, v in opp_quota_dict.items():
        opp_quota[int(s)] = float(v)
    
    # 初始化优化参数
    tau_param = init_tau.clone().detach().requires_grad_(True)
    quota_param = init_quota.clone().detach().requires_grad_(True)
    
    opt = _make_optimizer([tau_param, quota_param], config.opt_config)
    
    max_t = torch.tensor(float(config.constraints.max_tariff), device=DEFAULT_DEVICE)
    min_q = float(config.constraints.min_quota)
    
    for _ in range(int(config.opt_config.iterations)):
        opt.zero_grad(set_to_none=True)
        
        sim_fork = sim.fork_differentiable()
        
        # 裁剪参数到约束范围
        clamped_tau = torch.clamp(tau_param, min=0.0, max=max_t)
        clamped_quota = torch.clamp(quota_param, min=min_q, max=1.0)
        final_tau = torch.where(active_mask, clamped_tau, init_tau)
        final_quota = torch.where(active_mask, clamped_quota, init_quota)
        
        # 应用双方策略
        _apply_policy_differentiable(sim_fork, country=country, tau_rate=final_tau, quota_mult=final_quota)
        _apply_policy_differentiable(sim_fork, country=opponent, tau_rate=opp_tau, quota_mult=opp_quota)
        
        # 运行前瞻仿真
        sim_fork.run(lookahead)
        
        # 计算目标（只优化当前国家）
        J = objective.compute(sim_fork.history, country, opponent)
        loss = -J  # 最大化目标
        
        # 梯度更新
        grad = torch.autograd.grad(loss, [tau_param, quota_param])
        tau_param.grad = grad[0]
        quota_param.grad = grad[1]
        opt.step()
    
    # 获取最终策略
    tau_final = torch.clamp(tau_param.detach(), min=0.0, max=max_t)
    quota_final = torch.clamp(quota_param.detach(), min=min_q, max=1.0)
    tau_vec = torch.where(active_mask, tau_final, init_tau)
    quota_vec = torch.where(active_mask, quota_final, init_quota)
    
    # 预测效用
    sim_pred = sim.fork_differentiable()
    _apply_policy_differentiable(sim_pred, country=country, tau_rate=tau_vec, quota_mult=quota_vec)
    _apply_policy_differentiable(sim_pred, country=opponent, tau_rate=opp_tau, quota_mult=opp_quota)
    sim_pred.run(lookahead)
    J_pred = float(objective.compute(sim_pred.history, country, opponent).detach().cpu().item())
    
    policies = {
        "tariff": _policy_dict_from_vector(tau_vec, config.constraints.active_sectors),
        "quota": _policy_dict_from_vector(quota_vec, config.constraints.active_sectors),
    }
    info = {"J_pred": J_pred}
    return policies, info


def _create_llm_client(config: LLMGameConfig):
    """创建 LLM 客户端。"""
    logger.info(f"Using OpenAI-compatible client: preset={config.llm.preset}, model={config.llm.model}")
    return OpenAIClient(
        model=config.llm.model,
        preset=config.llm.preset,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )


def run_llm_experiment(config: LLMGameConfig) -> TwoCountryDynamicSimulator:
    """运行 LLM 博弈实验。"""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Simulator...")
    sim = bootstrap_simulator(
        create_symmetric_parameters(),
        theta_price=float(config.theta_price),
        normalize_gap_by_supply=bool(config.normalize_gap_by_supply),
    )

    # 预热
    sim.run(int(config.warmup_periods))
    if config.trigger is not None:
        _apply_trigger(sim, config.trigger)
    if int(config.trigger_settle_periods) > 0:
        sim.run(int(config.trigger_settle_periods))

    # 创建 LLM 客户端和代理
    llm_client = _create_llm_client(config)
    agent = LLMPolicyAgent(
        llm_client=llm_client,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    policy_history: List[Dict[str, Any]] = []
    timing_stats: List[float] = []
    total_start = time.perf_counter()

    objective = DifferentiableObjective(config.objective)
    n_sectors = int(sim.params.home.alpha.shape[0])

    for r in range(int(config.rounds)):
        logger.info(f"=== ROUND {r+1} ===")

        # 获取当前策略状态
        prev_tau_H = _current_tariff_rate(sim, "H")
        prev_tau_F = _current_tariff_rate(sim, "F")
        prev_quota_H = _current_quota_multiplier(sim, "H")
        prev_quota_F = _current_quota_multiplier(sim, "F")

        round_rec: Dict[str, Any] = {
            "round": r + 1,
            "prev": {
                "H": {
                    "tariff": _policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors),
                },
                "F": {
                    "tariff": _policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors),
                },
            },
        }

        t0 = time.perf_counter()
        decisions = {"H": {}, "F": {}}
        reasoning_log = {"H": "", "F": ""}

        # 为每个国家生成决策
        for country in ["H", "F"]:
            opponent = "F" if country == "H" else "H"
            
            current_tau = prev_tau_H if country == "H" else prev_tau_F
            current_quota = prev_quota_H if country == "H" else prev_quota_F
            opp_tau = prev_tau_F if country == "H" else prev_tau_H
            opp_quota = prev_quota_F if country == "H" else prev_quota_H

            if config.llm_plays == "both" or config.llm_plays == country:
                # 使用 LLM 决策
                decision = agent.decide(
                    sim=sim,
                    country=country,
                    round_num=r + 1,
                    opponent_prev_tariff=_policy_dict_from_vector(opp_tau, config.constraints.active_sectors),
                    opponent_prev_quota=_policy_dict_from_vector(opp_quota, config.constraints.active_sectors),
                    current_tariff=_policy_dict_from_vector(current_tau, config.constraints.active_sectors),
                    current_quota=_policy_dict_from_vector(current_quota, config.constraints.active_sectors),
                    active_sectors=list(config.constraints.active_sectors),
                    max_tariff=config.constraints.max_tariff,
                    min_quota=config.constraints.min_quota,
                    reciprocal_coeff=config.constraints.reciprocal_coeff,
                )
                decisions[country] = {
                    "tariff": decision.tariff,
                    "quota": decision.quota,
                    # 完整 I/O 记录
                    "system_prompt": decision.system_prompt,
                    "user_prompt": decision.user_prompt,
                    "raw_response": decision.raw_response,
                    "reasoning_content": decision.llm_reasoning_content,
                    "token_usage": decision.token_usage,
                }
                reasoning_log[country] = f"[LLM] {decision.reasoning}"
                logger.info(f"[{country}] LLM reasoning: {decision.reasoning}")
            elif config.non_llm_strategy == "gradient":
                # 使用梯度优化
                # 【重要】同步博弈：只使用对手的历史策略，不能看到对手本轮决策
                opponent_policy = {
                    "tariff": _policy_dict_from_vector(opp_tau, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(opp_quota, config.constraints.active_sectors),
                }
                
                grad_policy, grad_info = _optimize_single_country_gradient(
                    sim=sim,
                    country=country,
                    opponent_policy=opponent_policy,
                    config=config,
                )
                decisions[country] = {
                    "tariff": grad_policy["tariff"],
                    "quota": grad_policy["quota"],
                }
                reasoning_log[country] = f"[Gradient] J_pred={grad_info['J_pred']:.4f}"
                logger.info(f"[{country}] Gradient optimization: J_pred={grad_info['J_pred']:.4f}")
            else:
                # 保持当前策略不变
                decisions[country] = {
                    "tariff": _policy_dict_from_vector(current_tau, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(current_quota, config.constraints.active_sectors),
                }
                reasoning_log[country] = "[Fixed] Keeping current policy"

        opt_elapsed = time.perf_counter() - t0
        timing_stats.append(opt_elapsed)

        tau_H, quota_H = decisions["H"]["tariff"], decisions["H"]["quota"]
        tau_F, quota_F = decisions["F"]["tariff"], decisions["F"]["quota"]
        
        round_rec["decision"] = {
            "H": {"tariff": tau_H, "quota": quota_H},
            "F": {"tariff": tau_F, "quota": quota_F},
        }
        round_rec["reasoning"] = reasoning_log
        round_rec["llm_time_s"] = float(opt_elapsed)
        
        # 记录完整的 LLM I/O（用于复盘）
        round_rec["llm_io"] = {}
        for country in ["H", "F"]:
            d = decisions[country]
            if "system_prompt" in d:  # LLM 决策才有这些字段
                round_rec["llm_io"][country] = {
                    "system_prompt": d.get("system_prompt", ""),
                    "user_prompt": d.get("user_prompt", ""),
                    "raw_response": d.get("raw_response", ""),
                    "reasoning_content": d.get("reasoning_content"),
                    "token_usage": d.get("token_usage"),
                }

        logger.info(f"[Round {r+1}] H tariff={tau_H} quota={quota_H}")
        logger.info(f"[Round {r+1}] F tariff={tau_F} quota={quota_F}")

        # 应用策略
        if tau_H:
            sim.apply_import_tariff("H", tau_H, note=f"R{r+1} LLM Decision")
        if quota_H:
            sim.apply_export_control("H", quota_H, note=f"R{r+1} LLM Decision")
        if tau_F:
            sim.apply_import_tariff("F", tau_F, note=f"R{r+1} LLM Decision")
        if quota_F:
            sim.apply_export_control("F", quota_F, note=f"R{r+1} LLM Decision")

        # 运行仿真
        start_idx = len(sim.history["H"]) - 1
        sim.run(int(config.decision_interval))

        # 计算效用
        realized_hist = _history_slice(sim, start_idx)
        payoff_H = float(objective.compute(realized_hist, "H", "F").detach().cpu().item())
        payoff_F = float(objective.compute(realized_hist, "F", "H").detach().cpu().item())
        round_rec["payoff"] = {"H": payoff_H, "F": payoff_F}

        # 记录指标
        summary = sim.summarize_history(base_period_idx=int(config.warmup_periods))
        round_rec["metrics"] = {
            "income_H": float(summary["H"]["income"][-1]),
            "trade_balance_H": float(summary["H"]["trade_balance_val"][-1]),
            "price_mean_H": float(summary["H"]["price_mean"][-1]),
            "income_F": float(summary["F"]["income"][-1]),
            "trade_balance_F": float(summary["F"]["trade_balance_val"][-1]),
            "price_mean_F": float(summary["F"]["price_mean"][-1]),
        }

        policy_history.append(round_rec)

    total_elapsed = time.perf_counter() - total_start

    # 保存结果
    out_path = Path(config.output_dir) / f"{config.name}_policies.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "name": config.name,
                    "rounds": int(config.rounds),
                    "theta_price": float(config.theta_price),
                    "normalize_gap_by_supply": bool(config.normalize_gap_by_supply),
                    "decision_interval": int(config.decision_interval),
                    "lookahead_periods": int(config.lookahead_periods),
                    "warmup_periods": int(config.warmup_periods),
                    "trigger_settle_periods": int(config.trigger_settle_periods),
                    "trigger": (
                        None
                        if config.trigger is None
                        else {"country": config.trigger.country, "tariff": config.trigger.tariff, "quota": config.trigger.quota}
                    ),
                    "llm": {
                        "model": config.llm.model,
                        "preset": config.llm.preset,
                        "temperature": (None if config.llm.temperature is None else float(config.llm.temperature)),
                        "max_tokens": (None if config.llm.max_tokens is None else int(config.llm.max_tokens)),
                        "llm_plays": config.llm_plays,
                    },
                    "non_llm_strategy": str(config.non_llm_strategy),
                    "opt": {
                        "optimizer": config.opt_config.optimizer,
                        "learning_rate": float(config.opt_config.learning_rate),
                        "iterations": int(config.opt_config.iterations),
                    },
                    "constraints": {
                        "active_sectors": list(config.constraints.active_sectors),
                        "reciprocal_coeff": float(config.constraints.reciprocal_coeff),
                        "max_tariff": float(config.constraints.max_tariff),
                        "min_quota": float(config.constraints.min_quota),
                    },
                    "objective": {"type": config.objective.type, "weights": tuple(float(x) for x in config.objective.weights)},
                },
                "policies": policy_history,
                "timing": {
                    "total_s": float(total_elapsed),
                    "per_round_s": [float(x) for x in timing_stats],
                    "avg_per_round_s": float(sum(timing_stats) / len(timing_stats)) if timing_stats else 0.0,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved policy history to: {out_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("LLM 博弈策略决策摘要")
    print("=" * 80)
    for rec in policy_history:
        print(f"\n轮次 {rec['round']}:")
        print(f"  决策(H): {rec['decision']['H']}")
        print(f"  推理(H): {rec['reasoning']['H']}")
        print(f"  决策(F): {rec['decision']['F']}")
        print(f"  推理(F): {rec['reasoning']['F']}")
        print(f"  事后效用: H={rec['payoff']['H']:.4f}, F={rec['payoff']['F']:.4f}")
        print(f"  LLM 耗时: {rec['llm_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  LLM 总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮 LLM: {sum(timing_stats) / len(timing_stats):.2f}s")
    print("=" * 80 + "\n")

    # 生成可视化
    if config.plot:
        plot_path = Path(config.output_dir) / f"{config.name}_analysis.png"
        plot_game_analysis(
            sim.summarize_history(base_period_idx=int(config.warmup_periods)),
            sim.policy_events,
            save_path=str(plot_path),
        )
        logger.info(f"Plot saved to: {plot_path}")
        
        gap_plot_path = Path(config.output_dir) / f"{config.name}_supply_demand_gap.png"
        plot_supply_demand_gap(sim, save_path=str(gap_plot_path))
        logger.info(f"Supply-demand gap plot saved to: {gap_plot_path}")

    return sim


if __name__ == "__main__":
    # 例子：H 为 LLM 代理，F 为“计算策略”（梯度优化 best-response）。
    #
    # 使用 DeepSeek API 时，请先：
    #   export DEEPSEEK_API_KEY="your-api-key"
    #
    # 想让两方都用 LLM：将 llm_plays 改为 "both"
    # 想让 F 为 LLM、H 为计算方：将 llm_plays 改为 "F"
    cfg = LLMGameConfig(
        name="llm_vs_gradient",
        # NOTE (experiment): 这里打开按供给归一的价格调整公式（归一化后的公式 16）。
        normalize_gap_by_supply=True,
        # NOTE (experiment): 启用归一化后，原先的 theta_price=12500 会导致价格指数爆炸（exp 溢出）。
        theta_price=0.1,
        rounds=10,
        decision_interval=10,
        lookahead_periods=12,
        warmup_periods=1000,
        trigger_settle_periods=0,
        trigger=TriggerConfig(country="F", tariff={4: 0.5}),
        constraints=ConstraintsConfig(active_sectors=[2, 3], reciprocal_coeff=0, max_tariff=1.0, min_quota=0.0),
        objective=ObjectiveConfig(type="standard", weights=(1.0, 1.0, 1.0)),
        llm=LLMConfig(
            preset="deepseek",
            model="deepseek-chat",
        ),
        llm_plays="H",
        non_llm_strategy="gradient",
        opt_config=OptimizationConfig(learning_rate=0.01, iterations=200, optimizer="Adam",multi_start=8, start_strategy="noisy_current", select="sum"),
        plot=True,
        output_dir="results_0117_llmH_strategyF_F",
    )
    run_llm_experiment(cfg)
