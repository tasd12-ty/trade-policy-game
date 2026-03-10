"""LLM 博弈实验（两国同回合同步决策，由 LLM 生成策略）。

与 grad_game.py 平行，使用 LLM 替代梯度优化生成策略。
"""

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model import (
    bootstrap_simulator,
    TwoCountryDynamicSimulator,
    create_symmetric_parameters,
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
    _apply_policy,
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
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 2048


@dataclass(frozen=True)
class LLMGameConfig:
    """LLM 博弈实验配置。"""
    name: str = "llm_game"
    theta_price: float = 0.1
    normalize_gap_by_supply: bool = True
    rounds: int = 10
    decision_interval: int = 10
    lookahead_periods: int = 10
    warmup_periods: int = 1000
    trigger_settle_periods: int = 0
    trigger: Optional[TriggerConfig] = None
    constraints: ConstraintsConfig = ConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    llm: LLMConfig = LLMConfig()
    llm_plays: Literal["H", "F", "both"] = "both"
    non_llm_strategy: Literal["fixed", "search", "gradient"] = "fixed"
    opt_config: OptimizationConfig = OptimizationConfig()
    plot: bool = True
    output_dir: str = "results_llm"


def _optimize_single_country_search(
    sim: TwoCountryDynamicSimulator,
    country: Country,
    opponent_policy: Dict[str, Dict[int, float]],
    *,
    config: LLMGameConfig,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """为单个国家运行随机/邻域搜索，对手策略固定。"""
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    opponent: Country = "F" if country == "H" else "H"

    active_mask = np.zeros(n_sectors, dtype=bool)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True

    init_tau = _current_tariff_rate(sim, country)
    init_quota = _current_quota_multiplier(sim, country)

    opp_tau = np.zeros(n_sectors, dtype=float)
    opp_quota = np.ones(n_sectors, dtype=float)
    for s, v in opponent_policy["tariff"].items():
        opp_tau[int(s)] = float(v)
    for s, v in opponent_policy["quota"].items():
        opp_quota[int(s)] = float(v)

    max_t = float(config.constraints.max_tariff)
    min_q = float(config.constraints.min_quota)

    rng = np.random.default_rng(int(config.opt_config.seed))
    samples = max(int(config.opt_config.iterations), 1)

    best_tau = init_tau.copy()
    best_quota = init_quota.copy()
    best_J = float("-inf")

    for _ in range(samples):
        if config.opt_config.start_strategy == "current":
            cand_tau = init_tau.copy()
            cand_quota = init_quota.copy()
        else:
            if config.opt_config.start_strategy == "random":
                cand_tau = rng.random(init_tau.shape) * max_t
                cand_quota = min_q + (1.0 - min_q) * rng.random(init_quota.shape)
            elif config.opt_config.start_strategy == "noisy_current":
                sigma = float(max(config.opt_config.start_noise, 0.0))
                cand_tau = np.clip(init_tau + sigma * max_t * rng.standard_normal(init_tau.shape), 0.0, max_t)
                cand_quota = np.clip(init_quota + sigma * rng.standard_normal(init_quota.shape), min_q, 1.0)
            else:
                raise ValueError(f"Unknown start_strategy: {config.opt_config.start_strategy}")

        tau_vec = np.where(active_mask, cand_tau, init_tau)
        quota_vec = np.where(active_mask, cand_quota, init_quota)

        sim_fork = sim.fork()
        _apply_policy(sim_fork, country=country, tau_rate=tau_vec, quota_mult=quota_vec)
        _apply_policy(sim_fork, country=opponent, tau_rate=opp_tau, quota_mult=opp_quota)
        sim_fork.run(lookahead)

        J = float(objective.compute(sim_fork.history, country, opponent))
        if J > best_J:
            best_J = J
            best_tau = tau_vec
            best_quota = quota_vec

    policies = {
        "tariff": _policy_dict_from_vector(best_tau, config.constraints.active_sectors),
        "quota": _policy_dict_from_vector(best_quota, config.constraints.active_sectors),
    }
    info = {"J_pred": float(best_J)}
    return policies, info


def _create_llm_client(config: LLMGameConfig):
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

    sim.run(int(config.warmup_periods))
    if config.trigger is not None:
        _apply_trigger(sim, config.trigger)
    if int(config.trigger_settle_periods) > 0:
        sim.run(int(config.trigger_settle_periods))

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

    for r in range(int(config.rounds)):
        logger.info(f"=== ROUND {r+1} ===")

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
        round_rec["reasoning"] = {"H": "", "F": ""}

        t0 = time.perf_counter()

        policies: Dict[str, Dict[str, Dict[int, float]]] = {}
        pred: Dict[str, float] = {}

        # LLM 决策
        if config.llm_plays in ("H", "both"):
            decision_H = agent.decide(
                sim=sim,
                country="H",
                round_num=r + 1,
                opponent_prev_tariff=_policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                opponent_prev_quota=_policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors),
                current_tariff=_policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                current_quota=_policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors),
                active_sectors=config.constraints.active_sectors,
                max_tariff=config.constraints.max_tariff,
                min_quota=config.constraints.min_quota,
                reciprocal_coeff=config.constraints.reciprocal_coeff,
            )
            policies["H"] = {"tariff": decision_H.tariff, "quota": decision_H.quota}
            round_rec["llm_response_H"] = decision_H.raw_response
            round_rec["reasoning"]["H"] = decision_H.reasoning
        if config.llm_plays in ("F", "both"):
            decision_F = agent.decide(
                sim=sim,
                country="F",
                round_num=r + 1,
                opponent_prev_tariff=_policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                opponent_prev_quota=_policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors),
                current_tariff=_policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                current_quota=_policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors),
                active_sectors=config.constraints.active_sectors,
                max_tariff=config.constraints.max_tariff,
                min_quota=config.constraints.min_quota,
                reciprocal_coeff=config.constraints.reciprocal_coeff,
            )
            policies["F"] = {"tariff": decision_F.tariff, "quota": decision_F.quota}
            round_rec["llm_response_F"] = decision_F.raw_response
            round_rec["reasoning"]["F"] = decision_F.reasoning

        # 非 LLM 一方
        non_llm_strategy = config.non_llm_strategy
        if non_llm_strategy == "gradient":
            logger.info("non_llm_strategy=gradient 已弃用，自动改为 search")
            non_llm_strategy = "search"

        if non_llm_strategy == "fixed":
            if config.llm_plays == "H":
                policies.setdefault("F", {"tariff": _policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                                            "quota": _policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors)})
                round_rec["reasoning"]["F"] = "[Fixed] Keeping current policy"
            if config.llm_plays == "F":
                policies.setdefault("H", {"tariff": _policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                                            "quota": _policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors)})
                round_rec["reasoning"]["H"] = "[Fixed] Keeping current policy"
        elif non_llm_strategy == "search":
            if config.llm_plays == "H":
                opp_policy = policies.get("H")
                search_policy, info = _optimize_single_country_search(sim, "F", {"tariff": opp_policy["tariff"], "quota": opp_policy["quota"]}, config=config)
                policies.setdefault("F", search_policy)
                pred["J_F_pred"] = info.get("J_pred")
                round_rec["reasoning"]["F"] = f"[Search] J_pred={info.get('J_pred'):.4f}"
            if config.llm_plays == "F":
                opp_policy = policies.get("F")
                search_policy, info = _optimize_single_country_search(sim, "H", {"tariff": opp_policy["tariff"], "quota": opp_policy["quota"]}, config=config)
                policies.setdefault("H", search_policy)
                pred["J_H_pred"] = info.get("J_pred")
                round_rec["reasoning"]["H"] = f"[Search] J_pred={info.get('J_pred'):.4f}"
        else:
            raise ValueError(f"Unknown non_llm_strategy: {config.non_llm_strategy}")

        opt_elapsed = time.perf_counter() - t0
        timing_stats.append(opt_elapsed)

        tau_H, quota_H = policies["H"]["tariff"], policies["H"]["quota"]
        tau_F, quota_F = policies["F"]["tariff"], policies["F"]["quota"]
        round_rec["decision"] = {"H": {"tariff": tau_H, "quota": quota_H}, "F": {"tariff": tau_F, "quota": quota_F}}
        round_rec["predicted"] = pred
        round_rec["opt_time_s"] = float(opt_elapsed)

        logger.info(f"[Round {r+1}] H tariff={tau_H} quota={quota_H} | pred_J={pred.get('J_H_pred')}")
        logger.info(f"[Round {r+1}] F tariff={tau_F} quota={quota_F} | pred_J={pred.get('J_F_pred')}")

        if tau_H:
            sim.apply_import_tariff("H", tau_H, note=f"R{r+1} Decision")
        if quota_H:
            sim.apply_export_control("H", quota_H, note=f"R{r+1} Decision")
        if tau_F:
            sim.apply_import_tariff("F", tau_F, note=f"R{r+1} Decision")
        if quota_F:
            sim.apply_export_control("F", quota_F, note=f"R{r+1} Decision")

        start_idx = len(sim.history["H"]) - 1
        sim.run(int(config.decision_interval))

        realized_hist = _history_slice(sim, start_idx)
        payoff_H = float(objective.compute(realized_hist, "H", "F"))
        payoff_F = float(objective.compute(realized_hist, "F", "H"))
        round_rec["payoff"] = {"H": payoff_H, "F": payoff_F}

        summary = sim.summarize_history()
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
                        "multi_start": int(config.opt_config.multi_start),
                        "start_strategy": str(config.opt_config.start_strategy),
                        "start_noise": float(config.opt_config.start_noise),
                        "seed": int(config.opt_config.seed),
                        "select": str(config.opt_config.select),
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
        print(f"  决策耗时: {rec['opt_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  决策总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮决策: {sum(timing_stats) / len(timing_stats):.2f}s")
    print("=" * 80 + "\n")

    if config.plot:
        plot_path = Path(config.output_dir) / f"{config.name}_analysis.png"
        plot_game_analysis(sim.summarize_history(), sim.policy_events, save_path=str(plot_path))
        logger.info(f"Plot saved to: {plot_path}")
        gap_plot_path = Path(config.output_dir) / f"{config.name}_supply_demand_gap.png"
        plot_supply_demand_gap(sim, save_path=str(gap_plot_path))
        logger.info(f"Supply-demand gap plot saved to: {gap_plot_path}")

    return sim


if __name__ == "__main__":
    # 例子：H 为 LLM 代理，F 为搜索策略（非梯度）。
    #
    # 使用 DeepSeek API 时，请先：
    #   export DEEPSEEK_API_KEY="your-api-key"
    #
    # 想让两方都用 LLM：将 llm_plays 改为 "both"
    # 想让 F 为 LLM、H 为搜索方：将 llm_plays 改为 "F"
    cfg = LLMGameConfig(
        name="llm_vs_search",
        # NOTE (experiment): 这里打开按供给归一的价格调整公式（归一化后的公式 16）。
        normalize_gap_by_supply=True,
        # NOTE (experiment): 启用归一化后，原先的 theta_price=12500 会导致价格指数爆炸（exp 溢出）。
        theta_price=0.10,
        rounds=10,
        decision_interval=50,
        lookahead_periods=5,
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
        non_llm_strategy="search",
        opt_config=OptimizationConfig(iterations=500, multi_start=8, start_strategy="noisy_current", select="sum"),
        plot=True,
        output_dir="results_llmH_strategyF_F_50_5",
    )
    run_llm_experiment(cfg)
