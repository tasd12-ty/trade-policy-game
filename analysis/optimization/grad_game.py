"""梯度博弈实验（两国同回合同步决策）。

实现逻辑（简洁版）：
1) 初始化两国动态仿真器（静态均衡 + 预热）；
2) 施加一个初始 trigger 启动冲突；
3) 每回合 H/F 同时选择政策（关税率 tau、出口配额乘子 quota），并可基于对手上一回合政策形成对等反制约束；
4) 用 lookahead 仿真期数评估预期效用，采用 gradient play 同步更新两国决策变量；
5) 将两国当回合政策同时施加到真实仿真器，推进 decision_interval 期，并记录事后效用与关键指标。

运行：
    python analysis/optimization/grad_game.py
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Country = Literal["H", "F"]


@dataclass(frozen=True)
class TriggerConfig:
    """启动博弈的初始触发器（可选）。"""

    country: Country = "H"
    tariff: Dict[int, float] = field(default_factory=dict)  # {sector: rate}
    quota: Dict[int, float] = field(default_factory=dict)  # {sector: multiplier in [0,1]}


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.01
    iterations: int = 50
    optimizer: Literal["Adam", "SGD"] = "Adam"
    # Multi-start (batch) search: generate K initial points and optimize them in parallel,
    # then select one "best" trajectory according to `select`.
    multi_start: int = 1
    start_strategy: Literal["current", "noisy_current", "random"] = "current"
    start_noise: float = 0.05
    seed: int = 42
    # How to select the "best" trajectory among K after optimization:
    # - "sum": maximize (J_H + J_F)
    # - "min": maximize min(J_H, J_F)
    # - "H": maximize J_H only
    # - "F": maximize J_F only
    select: Literal["sum", "min", "H", "F"] = "sum"


@dataclass(frozen=True)
class ConstraintsConfig:
    active_sectors: List[int] = field(default_factory=lambda: [0, 1])
    reciprocal_coeff: float = 0.0
    max_tariff: float = 1.0
    min_quota: float = 0.0


@dataclass(frozen=True)
class ObjectiveConfig:
    type: Literal["standard", "relative"] = "standard"
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (收入, 贸易差额, 稳定性)


@dataclass(frozen=True)
class GameConfig:
    """博弈实验总配置（两国同步决策）。

    参数语义（建议用法）：
    - `rounds`：决策回合数。每回合包含：同步求解当回合政策 -> 同时施策 -> 推进 `decision_interval` 期 -> 记录事后效用。
    - `decision_interval`：决策间隔（真实世界推进期数）。例如 10 表示每次决策后向前跑 10 期才进入下一次决策。
    - `lookahead_periods`：lookahead 期数（预期效用窗口）。每次求解当回合政策时，用分叉的可微仿真器向前跑这么多期来评估目标。
      重要：它不需要等于 `decision_interval`；常见做法是 lookahead 更长（更“远视”）或更短（更快）。
    - `warmup_periods`：热启动/预热期数。用于让系统离开静态均衡附近的瞬态，进入更稳定的动态轨道。
    - `trigger`：初始触发器（关税/配额冲击），发生在预热之后，用于“启动”博弈过程。
    - `trigger_settle_periods`：触发器施加后，先推进多少期再开始第 1 回合。
      这样双方在第 1 回合决策时，看到的是“触发器造成的实际后果”而非刚施加的瞬间。
    - `objective`：目标函数配置；`type="standard"` 表示只关注本国效益（收入增长 + 贸易差额 + 价格稳定）。
    """
    name: str = "grad_simultaneous"
    # NOTE (experiment): 价格更新速度（对应 analysis/model/sim.py 里的 theta_price / τ）。
    theta_price: float = 12500.0
    # NOTE (experiment): 是否启用归一化后的公式 16（按供给归一）：(D-Y)/Y。
    # 用于科研对比：同一套博弈流程下切换不同价格调整公式，尽量不增加耦合度。
    normalize_gap_by_supply: bool = False
    rounds: int = 10
    decision_interval: int = 10
    lookahead_periods: int = 10
    warmup_periods: int = 10
    trigger_settle_periods: int = 10
    trigger: Optional[TriggerConfig] = None
    opt_config: OptimizationConfig = OptimizationConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    plot: bool = True
    output_dir: str = "results_grad"


class DifferentiableObjective:
    """可微目标函数：用仿真历史计算单国效用，保留梯度图以供优化。"""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.w_income, self.w_tb, self.w_stab = config.weights

    def compute(self, history: Dict[str, List[CountryState]], actor: str, opponent: str) -> torch.Tensor:
        """返回每个 batch 元素的效用。

        - 非 batch 仿真：返回 0-d tensor（标量）。
        - batch 仿真：返回 shape=(B,) 的 1-d tensor，每个元素对应一条并行轨迹。
        """
        states_self = history[actor]
        states_opp = history[opponent]

        def get_seq(states, attr):
            return torch.stack([getattr(s, attr) for s in states[1:]])

        income_self = get_seq(states_self, "income")
        income_self_0 = states_self[0].income
        growth_self = (income_self / (income_self_0 + 1e-6)) - 1.0  # (T,) or (T,B)
        score_income_self = growth_self.mean(dim=0)

        def compute_tb(s):
            exp_val = (s.export_actual * s.price).sum(dim=-1)
            imp_val = (s.imp_price * (s.X_imp.sum(dim=-2) + s.C_imp)).sum(dim=-1)
            return exp_val - imp_val

        tb_self_seq = torch.stack([compute_tb(s) for s in states_self[1:]])  # (T,) or (T,B)
        scale = income_self_0 + 1.0
        score_tb_self = (tb_self_seq / scale).mean(dim=0)

        def get_price_idx(s, p0):
            return (s.price / (p0 + 1e-6)).mean(dim=-1)

        p0 = states_self[0].price
        p_idx_seq = torch.stack([get_price_idx(s, p0) for s in states_self[1:]])  # (T,) or (T,B)
        score_stab_self = -torch.std(p_idx_seq, dim=0)

        j_std = self.w_income * score_income_self + self.w_tb * score_tb_self + self.w_stab * score_stab_self

        if self.config.type == "relative":
            income_opp = get_seq(states_opp, "income")
            income_opp_0 = states_opp[0].income
            growth_opp = (income_opp / (income_opp_0 + 1e-6)) - 1.0
            score_income_opp = growth_opp.mean(dim=0)
            tb_opp_seq = torch.stack([compute_tb(s) for s in states_opp[1:]])
            score_tb_opp = (tb_opp_seq / (income_opp_0 + 1.0)).mean(dim=0)
            return (
                self.w_income * (score_income_self - score_income_opp)
                + self.w_tb * (score_tb_self - score_tb_opp)
                + self.w_stab * score_stab_self
            )

        return j_std


def _make_optimizer(params: List[torch.Tensor], cfg: OptimizationConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "Adam":
        return torch.optim.Adam(params, lr=float(cfg.learning_rate))
    if cfg.optimizer == "SGD":
        return torch.optim.SGD(params, lr=float(cfg.learning_rate))
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _randn_like(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """torch.randn_like 的兼容封装：不同 torch 版本对 generator 参数支持不一致。"""
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def _current_tariff_rate(sim: TwoCountryDynamicSimulator, country: Country) -> torch.Tensor:
    if country == "H":
        mult = sim.home_import_multiplier
    else:
        mult = sim.foreign_import_multiplier
    base = sim.baseline_import_multiplier[country]
    return (mult / (base + 1e-6) - 1.0).detach().clone().view(-1)


def _current_quota_multiplier(sim: TwoCountryDynamicSimulator, country: Country) -> torch.Tensor:
    return sim.export_multiplier[country].detach().clone().view(-1)


def _policy_dict_from_vector(x: torch.Tensor, active_sectors: List[int]) -> Dict[int, float]:
    x = x.detach().view(-1).cpu()
    return {int(s): float(x[int(s)].item()) for s in active_sectors}


def _apply_policy_differentiable(sim_fork: TwoCountryDynamicSimulator, *, country: Country, tau_rate: torch.Tensor, quota_mult: torch.Tensor) -> None:
    if country == "H":
        sim_fork.home_import_multiplier = sim_fork.baseline_import_multiplier["H"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["H"] = quota_mult
        sim_fork._update_export_base("H")
    else:
        sim_fork.foreign_import_multiplier = sim_fork.baseline_import_multiplier["F"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["F"] = quota_mult
        sim_fork._update_export_base("F")


def _history_slice(sim: TwoCountryDynamicSimulator, start_idx: int) -> Dict[str, List[CountryState]]:
    return {"H": list(sim.history["H"][start_idx:]), "F": list(sim.history["F"][start_idx:])}


def _apply_trigger(sim: TwoCountryDynamicSimulator, trigger: TriggerConfig) -> None:
    if trigger.tariff:
        sim.apply_import_tariff(trigger.country, dict(trigger.tariff), note="Trigger: tariff")
    if trigger.quota:
        sim.apply_export_control(trigger.country, dict(trigger.quota), note="Trigger: quota")


def _optimize_simultaneous_gradient_play(
    sim: TwoCountryDynamicSimulator,
    *,
    config: GameConfig,
    prev_tau_H: torch.Tensor,
    prev_tau_F: torch.Tensor,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """同一回合内两国同时做 gradient play 得到当回合策略。"""
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    K = max(int(config.opt_config.multi_start), 1)

    active_mask = torch.zeros(n_sectors, dtype=torch.bool, device=DEFAULT_DEVICE)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True

    init_tau_H = _current_tariff_rate(sim, "H").to(DEFAULT_DEVICE)
    init_tau_F = _current_tariff_rate(sim, "F").to(DEFAULT_DEVICE)
    init_quota_H = _current_quota_multiplier(sim, "H").to(DEFAULT_DEVICE)
    init_quota_F = _current_quota_multiplier(sim, "F").to(DEFAULT_DEVICE)

    if K == 1:
        tau_H_param = init_tau_H.clone().detach().requires_grad_(True)
        tau_F_param = init_tau_F.clone().detach().requires_grad_(True)
        quota_H_param = init_quota_H.clone().detach().requires_grad_(True)
        quota_F_param = init_quota_F.clone().detach().requires_grad_(True)
    else:
        # Batch parameters: shape (K, n)
        gen = torch.Generator(device=DEFAULT_DEVICE)
        gen.manual_seed(int(config.opt_config.seed))

        def _repeat(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, -1).repeat(K, 1)

        tau_H_param = _repeat(init_tau_H).detach()
        tau_F_param = _repeat(init_tau_F).detach()
        quota_H_param = _repeat(init_quota_H).detach()
        quota_F_param = _repeat(init_quota_F).detach()

        max_t = float(config.constraints.max_tariff)
        min_q = float(config.constraints.min_quota)

        def _lb(prev_opp: torch.Tensor) -> torch.Tensor:
            if float(config.constraints.reciprocal_coeff) <= 0:
                return torch.zeros_like(prev_opp, device=DEFAULT_DEVICE).view(1, -1).repeat(K, 1)
            lb = float(config.constraints.reciprocal_coeff) * prev_opp.to(DEFAULT_DEVICE).view(1, -1).repeat(K, 1)
            return torch.clamp(lb, 0.0, max_t)

        lb_H = _lb(prev_tau_F)
        lb_F = _lb(prev_tau_H)
        ub_H = torch.full_like(lb_H, max_t)
        ub_F = torch.full_like(lb_F, max_t)

        strat = config.opt_config.start_strategy
        if strat == "random":
            tau_H_param = lb_H + (max_t - lb_H) * torch.rand((K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
            tau_F_param = lb_F + (max_t - lb_F) * torch.rand((K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
            quota_H_param = min_q + (1.0 - min_q) * torch.rand((K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
            quota_F_param = min_q + (1.0 - min_q) * torch.rand((K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
        elif strat == "noisy_current":
            sigma = float(max(config.opt_config.start_noise, 0.0))
            # Noise scaled by variable ranges; tau range depends on reciprocal lower bound.
            tau_H_width = torch.clamp(max_t - lb_H, min=1e-6)
            tau_F_width = torch.clamp(max_t - lb_F, min=1e-6)
            tau_H_param = torch.clamp(
                tau_H_param + sigma * tau_H_width * _randn_like(tau_H_param, generator=gen), min=lb_H, max=ub_H
            )
            tau_F_param = torch.clamp(
                tau_F_param + sigma * tau_F_width * _randn_like(tau_F_param, generator=gen), min=lb_F, max=ub_F
            )
            q_width = max(1.0 - min_q, 1e-6)
            quota_H_param = torch.clamp(quota_H_param + sigma * q_width * _randn_like(quota_H_param, generator=gen), min=min_q, max=1.0)
            quota_F_param = torch.clamp(quota_F_param + sigma * q_width * _randn_like(quota_F_param, generator=gen), min=min_q, max=1.0)
        elif strat == "current":
            pass
        else:
            raise ValueError(f"Unknown start_strategy: {strat}")

        tau_H_param.requires_grad_(True)
        tau_F_param.requires_grad_(True)
        quota_H_param.requires_grad_(True)
        quota_F_param.requires_grad_(True)

    opt_H = _make_optimizer([tau_H_param, quota_H_param], config.opt_config)
    opt_F = _make_optimizer([tau_F_param, quota_F_param], config.opt_config)

    max_t = torch.tensor(float(config.constraints.max_tariff), device=DEFAULT_DEVICE)
    min_q = float(config.constraints.min_quota)

    def lower_bound_tau(prev_opp: torch.Tensor) -> torch.Tensor:
        if float(config.constraints.reciprocal_coeff) <= 0:
            return torch.zeros(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
        lb = float(config.constraints.reciprocal_coeff) * prev_opp.to(DEFAULT_DEVICE)
        return torch.clamp(lb, 0.0, float(config.constraints.max_tariff))

    lb_tau_H = lower_bound_tau(prev_tau_F)
    lb_tau_F = lower_bound_tau(prev_tau_H)

    for _ in range(int(config.opt_config.iterations)):
        opt_H.zero_grad(set_to_none=True)
        opt_F.zero_grad(set_to_none=True)

        sim_fork = sim.fork_differentiable()
        if K > 1:
            # Expand snapshot state to batch dimension so we can simulate K trajectories in parallel.
            sim_fork.home_state = sim_fork.home_state.ensure_batch(K)
            sim_fork.foreign_state = sim_fork.foreign_state.ensure_batch(K)
            sim_fork.batch_size = K
            sim_fork.history = {"H": [sim_fork.home_state], "F": [sim_fork.foreign_state]}

        clamped_tau_H = torch.clamp(tau_H_param, min=lb_tau_H, max=max_t)
        clamped_tau_F = torch.clamp(tau_F_param, min=lb_tau_F, max=max_t)
        final_tau_H = torch.where(active_mask, clamped_tau_H, init_tau_H)
        final_tau_F = torch.where(active_mask, clamped_tau_F, init_tau_F)

        clamped_quota_H = torch.clamp(quota_H_param, min=min_q, max=1.0)
        clamped_quota_F = torch.clamp(quota_F_param, min=min_q, max=1.0)
        final_quota_H = torch.where(active_mask, clamped_quota_H, init_quota_H)
        final_quota_F = torch.where(active_mask, clamped_quota_F, init_quota_F)

        _apply_policy_differentiable(sim_fork, country="H", tau_rate=final_tau_H, quota_mult=final_quota_H)
        _apply_policy_differentiable(sim_fork, country="F", tau_rate=final_tau_F, quota_mult=final_quota_F)

        sim_fork.run(lookahead)

        J_H = objective.compute(sim_fork.history, "H", "F")
        J_F = objective.compute(sim_fork.history, "F", "H")
        # Make scalars for gradient-play updates; with batch, this sums independent trajectories.
        J_H_total = J_H.sum() if J_H.dim() > 0 else J_H
        J_F_total = J_F.sum() if J_F.dim() > 0 else J_F

        g_tau_H, g_quota_H = torch.autograd.grad(-J_H_total, [tau_H_param, quota_H_param], retain_graph=True)
        g_tau_F, g_quota_F = torch.autograd.grad(-J_F_total, [tau_F_param, quota_F_param], retain_graph=False)

        tau_H_param.grad = g_tau_H
        quota_H_param.grad = g_quota_H
        tau_F_param.grad = g_tau_F
        quota_F_param.grad = g_quota_F

        opt_H.step()
        opt_F.step()

    tau_H_final = torch.clamp(tau_H_param.detach(), min=lb_tau_H, max=max_t)
    tau_F_final = torch.clamp(tau_F_param.detach(), min=lb_tau_F, max=max_t)
    quota_H_final = torch.clamp(quota_H_param.detach(), min=min_q, max=1.0)
    quota_F_final = torch.clamp(quota_F_param.detach(), min=min_q, max=1.0)

    tau_H_vec = torch.where(active_mask, tau_H_final, init_tau_H)
    tau_F_vec = torch.where(active_mask, tau_F_final, init_tau_F)
    quota_H_vec = torch.where(active_mask, quota_H_final, init_quota_H)
    quota_F_vec = torch.where(active_mask, quota_F_final, init_quota_F)

    sim_pred = sim.fork_differentiable()
    if K > 1:
        sim_pred.home_state = sim_pred.home_state.ensure_batch(K)
        sim_pred.foreign_state = sim_pred.foreign_state.ensure_batch(K)
        sim_pred.batch_size = K
        sim_pred.history = {"H": [sim_pred.home_state], "F": [sim_pred.foreign_state]}
    _apply_policy_differentiable(sim_pred, country="H", tau_rate=tau_H_vec, quota_mult=quota_H_vec)
    _apply_policy_differentiable(sim_pred, country="F", tau_rate=tau_F_vec, quota_mult=quota_F_vec)
    sim_pred.run(lookahead)
    J_H_pred_all = objective.compute(sim_pred.history, "H", "F").detach()
    J_F_pred_all = objective.compute(sim_pred.history, "F", "H").detach()

    if K == 1:
        best_idx = 0
        J_H_pred = float(J_H_pred_all.cpu().item())
        J_F_pred = float(J_F_pred_all.cpu().item())
    else:
        if J_H_pred_all.dim() == 0 or J_F_pred_all.dim() == 0:
            raise RuntimeError("Expected batched objectives for multi_start > 1.")
        if config.opt_config.select == "sum":
            score = J_H_pred_all + J_F_pred_all
        elif config.opt_config.select == "min":
            score = torch.minimum(J_H_pred_all, J_F_pred_all)
        elif config.opt_config.select == "H":
            score = J_H_pred_all
        elif config.opt_config.select == "F":
            score = J_F_pred_all
        else:
            raise ValueError(f"Unknown select: {config.opt_config.select}")
        best_idx = int(torch.argmax(score).cpu().item())
        J_H_pred = float(J_H_pred_all[best_idx].cpu().item())
        J_F_pred = float(J_F_pred_all[best_idx].cpu().item())

        # Pick the best trajectory's policies.
        tau_H_vec = tau_H_vec[best_idx]
        tau_F_vec = tau_F_vec[best_idx]
        quota_H_vec = quota_H_vec[best_idx]
        quota_F_vec = quota_F_vec[best_idx]

    policies = {
        "H": {"tariff": _policy_dict_from_vector(tau_H_vec, config.constraints.active_sectors),
              "quota": _policy_dict_from_vector(quota_H_vec, config.constraints.active_sectors)},
        "F": {"tariff": _policy_dict_from_vector(tau_F_vec, config.constraints.active_sectors),
              "quota": _policy_dict_from_vector(quota_F_vec, config.constraints.active_sectors)},
    }
    info = {"J_H_pred": J_H_pred, "J_F_pred": J_F_pred, "multi_start": K, "best_idx": int(best_idx)}
    return policies, info


def run_grad_experiment(config: GameConfig) -> TwoCountryDynamicSimulator:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Simulator...")
    # NOTE (experiment): 通过 config 同时控制 theta_price 与归一化公式开关，便于做 A/B 对比实验。
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
                "H": {"tariff": _policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                      "quota": _policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors)},
                "F": {"tariff": _policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                      "quota": _policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors)},
            },
        }

        t0 = time.perf_counter()
        policies, pred = _optimize_simultaneous_gradient_play(sim, config=config, prev_tau_H=prev_tau_H, prev_tau_F=prev_tau_F)
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
        payoff_H = float(objective.compute(realized_hist, "H", "F").detach().cpu().item())
        payoff_F = float(objective.compute(realized_hist, "F", "H").detach().cpu().item())
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
                    "opt": {"optimizer": config.opt_config.optimizer, "learning_rate": config.opt_config.learning_rate, "iterations": config.opt_config.iterations},
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
    print("策略决策摘要")
    print("=" * 80)
    for rec in policy_history:
        print(f"\n轮次 {rec['round']}:")
        print(f"  决策(H): {rec['decision']['H']}")
        print(f"  决策(F): {rec['decision']['F']}")
        print(f"  预测效用: {rec['predicted']}")
        print(f"  事后效用: {rec['payoff']}")
        print(f"  优化耗时: {rec['opt_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  优化总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮优化: {sum(timing_stats) / len(timing_stats):.2f}s")
    print("=" * 80 + "\n")

    if config.plot:
        plot_path = Path(config.output_dir) / f"{config.name}_analysis.png"
        plot_game_analysis(sim.summarize_history(), sim.policy_events, save_path=str(plot_path))
        logger.info(f"Plot saved to: {plot_path}")
        
        # 供求之差可视化
        gap_plot_path = Path(config.output_dir) / f"{config.name}_supply_demand_gap.png"
        plot_supply_demand_gap(sim, save_path=str(gap_plot_path))
        logger.info(f"Supply-demand gap plot saved to: {gap_plot_path}")

    return sim


if __name__ == "__main__":
    # 例子：决策间隔 10 期、lookahead 15 期、热启动 200 期；
    # trigger 在热启动后发生；目标函数为 standard（只关注本国效益）。
    cfg = GameConfig(
        name="grad_simultaneous_trigger",
        # NOTE (experiment): 这里打开按供给归一的价格调整公式（归一化后的公式 16）。
        normalize_gap_by_supply=True,
        # NOTE (experiment): 启用归一化后，原先的 theta_price=12500 会导致价格指数爆炸（exp 溢出）。
        # 需要重新标定；这里给一个温和的起点，便于先跑通实验流程。
        theta_price=0.1,
        rounds=10,
        decision_interval=10,
        lookahead_periods=5,       
        warmup_periods=1000,
        trigger_settle_periods=0,
        trigger=TriggerConfig(country="H", tariff={4: 0.5}),
        opt_config=OptimizationConfig(learning_rate=0.01, iterations=200, optimizer="Adam",multi_start=8, start_strategy="noisy_current", select="sum"),
        constraints=ConstraintsConfig(active_sectors=[2, 3], reciprocal_coeff=0, max_tariff=1.0, min_quota=0.0),
        objective=ObjectiveConfig(type="standard", weights=(1.0, 1.0, 1.0)),
        plot=True,
        output_dir="results_grad",
    )
    run_grad_experiment(cfg)
