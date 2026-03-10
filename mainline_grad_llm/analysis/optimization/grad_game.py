"""非梯度博弈实验（两国同回合同步决策）。

实现逻辑：
1) 初始化两国动态仿真器（静态均衡 + 预热）；
2) 施加一个初始 trigger 启动冲突；
3) 每回合 H/F 同时选择政策（关税率 tau、出口配额 multiplier），并可基于对手上一回合政策形成对等反制约束；
4) 用 lookahead 仿真期数评估预期效用，采用随机/邻域搜索选择政策；
5) 将两国当回合政策同时施加到真实仿真器，推进 decision_interval 期，并记录事后效用与关键指标。
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
    CountryState,
)
from analysis.optimization.plotting import plot_game_analysis, plot_supply_demand_gap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Country = Literal["H", "F"]


@dataclass(frozen=True)
class TriggerConfig:
    """启动博弈的初始触发器（可选）。"""

    country: Country = "H"
    tariff: Dict[int, float] = field(default_factory=dict)
    quota: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.01
    iterations: int = 50
    optimizer: Literal["Adam", "SGD"] = "Adam"
    multi_start: int = 1
    start_strategy: Literal["current", "noisy_current", "random"] = "current"
    start_noise: float = 0.05
    seed: int = 42
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
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class GameConfig:
    name: str = "grad_simultaneous"
    theta_price: float = 12500.0
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
    """目标函数：用仿真历史计算单国效用（非梯度）。"""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.w_income, self.w_tb, self.w_stab = config.weights

    def compute(self, history: Dict[str, List[CountryState]], actor: str, opponent: str) -> float:
        states_self = history[actor]
        states_opp = history[opponent]

        income_self = np.array([s.income for s in states_self[1:]], dtype=float)
        income_self_0 = float(states_self[0].income)
        growth_self = (income_self / (income_self_0 + 1e-6)) - 1.0
        score_income_self = float(growth_self.mean()) if growth_self.size else 0.0

        def compute_tb(s: CountryState) -> float:
            exp_val = float((s.export_actual * s.price).sum())
            imp_val = float((s.imp_price * (s.X_imp.sum(axis=0) + s.C_imp)).sum())
            return exp_val - imp_val

        tb_self_seq = np.array([compute_tb(s) for s in states_self[1:]], dtype=float)
        scale = income_self_0 + 1.0
        score_tb_self = float((tb_self_seq / scale).mean()) if tb_self_seq.size else 0.0

        p0 = states_self[0].price
        p_idx_seq = np.array([np.mean(s.price / (p0 + 1e-6)) for s in states_self[1:]], dtype=float)
        score_stab_self = -float(np.std(p_idx_seq)) if p_idx_seq.size else 0.0

        j_std = self.w_income * score_income_self + self.w_tb * score_tb_self + self.w_stab * score_stab_self

        if self.config.type == "relative":
            income_opp = np.array([s.income for s in states_opp[1:]], dtype=float)
            income_opp_0 = float(states_opp[0].income)
            growth_opp = (income_opp / (income_opp_0 + 1e-6)) - 1.0
            score_income_opp = float(growth_opp.mean()) if growth_opp.size else 0.0
            tb_opp_seq = np.array([compute_tb(s) for s in states_opp[1:]], dtype=float)
            score_tb_opp = float((tb_opp_seq / (income_opp_0 + 1.0)).mean()) if tb_opp_seq.size else 0.0
            return (
                self.w_income * (score_income_self - score_income_opp)
                + self.w_tb * (score_tb_self - score_tb_opp)
                + self.w_stab * score_stab_self
            )

        return j_std


def _current_tariff_rate(sim: TwoCountryDynamicSimulator, country: Country) -> np.ndarray:
    if country == "H":
        mult = sim.home_import_multiplier
    else:
        mult = sim.foreign_import_multiplier
    base = sim.baseline_import_multiplier[country]
    return (mult / (base + 1e-6) - 1.0).copy()


def _current_quota_multiplier(sim: TwoCountryDynamicSimulator, country: Country) -> np.ndarray:
    return sim.export_multiplier[country].copy()


def _policy_dict_from_vector(x: np.ndarray, active_sectors: List[int]) -> Dict[int, float]:
    x = np.asarray(x).reshape(-1)
    return {int(s): float(x[int(s)]) for s in active_sectors}


def _apply_policy(sim_fork: TwoCountryDynamicSimulator, *, country: Country, tau_rate: np.ndarray, quota_mult: np.ndarray) -> None:
    if country == "H":
        sim_fork.home_import_multiplier = sim_fork.baseline_import_multiplier["H"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["H"] = quota_mult.copy()
        sim_fork._update_export_base("H")
    else:
        sim_fork.foreign_import_multiplier = sim_fork.baseline_import_multiplier["F"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["F"] = quota_mult.copy()
        sim_fork._update_export_base("F")


def _history_slice(sim: TwoCountryDynamicSimulator, start_idx: int) -> Dict[str, List[CountryState]]:
    return {"H": list(sim.history["H"][start_idx:]), "F": list(sim.history["F"][start_idx:])}


def _apply_trigger(sim: TwoCountryDynamicSimulator, trigger: TriggerConfig) -> None:
    if trigger.tariff:
        sim.apply_import_tariff(trigger.country, dict(trigger.tariff), note="Trigger: tariff")
    if trigger.quota:
        sim.apply_export_control(trigger.country, dict(trigger.quota), note="Trigger: quota")


def _sample_policy(
    rng: np.random.Generator,
    prev: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    strategy: str,
    noise: float,
) -> np.ndarray:
    if strategy == "current":
        return prev.copy()
    if strategy == "random":
        return lb + (ub - lb) * rng.random(prev.shape)
    if strategy == "noisy_current":
        sigma = float(max(noise, 0.0))
        width = np.maximum(ub - lb, 1e-6)
        proposal = prev + sigma * width * rng.standard_normal(prev.shape)
        return np.clip(proposal, lb, ub)
    raise ValueError(f"Unknown start_strategy: {strategy}")


def _optimize_simultaneous_search(
    sim: TwoCountryDynamicSimulator,
    *,
    config: GameConfig,
    prev_tau_H: np.ndarray,
    prev_tau_F: np.ndarray,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """同一回合内两国同时做随机/邻域搜索得到当回合策略。"""
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    K = max(int(config.opt_config.multi_start), 1)
    samples = max(int(config.opt_config.iterations), K)

    active_mask = np.zeros(n_sectors, dtype=bool)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True

    init_tau_H = _current_tariff_rate(sim, "H")
    init_tau_F = _current_tariff_rate(sim, "F")
    init_quota_H = _current_quota_multiplier(sim, "H")
    init_quota_F = _current_quota_multiplier(sim, "F")

    max_t = float(config.constraints.max_tariff)
    min_q = float(config.constraints.min_quota)

    def lower_bound_tau(prev_opp: np.ndarray) -> np.ndarray:
        if float(config.constraints.reciprocal_coeff) <= 0:
            return np.zeros_like(prev_opp)
        lb = float(config.constraints.reciprocal_coeff) * prev_opp
        return np.clip(lb, 0.0, max_t)

    lb_tau_H = lower_bound_tau(prev_tau_F)
    lb_tau_F = lower_bound_tau(prev_tau_H)
    ub_tau_H = np.full_like(lb_tau_H, max_t)
    ub_tau_F = np.full_like(lb_tau_F, max_t)

    rng = np.random.default_rng(int(config.opt_config.seed))

    best = None
    best_score = None
    best_J = (float("-inf"), float("-inf"))

    for _ in range(samples):
        tau_H_candidate = _sample_policy(rng, init_tau_H, lb_tau_H, ub_tau_H, config.opt_config.start_strategy, config.opt_config.start_noise)
        tau_F_candidate = _sample_policy(rng, init_tau_F, lb_tau_F, ub_tau_F, config.opt_config.start_strategy, config.opt_config.start_noise)
        quota_H_candidate = _sample_policy(rng, init_quota_H, np.full_like(init_quota_H, min_q), np.ones_like(init_quota_H), config.opt_config.start_strategy, config.opt_config.start_noise)
        quota_F_candidate = _sample_policy(rng, init_quota_F, np.full_like(init_quota_F, min_q), np.ones_like(init_quota_F), config.opt_config.start_strategy, config.opt_config.start_noise)

        final_tau_H = np.where(active_mask, tau_H_candidate, init_tau_H)
        final_tau_F = np.where(active_mask, tau_F_candidate, init_tau_F)
        final_quota_H = np.where(active_mask, quota_H_candidate, init_quota_H)
        final_quota_F = np.where(active_mask, quota_F_candidate, init_quota_F)

        sim_fork = sim.fork()
        _apply_policy(sim_fork, country="H", tau_rate=final_tau_H, quota_mult=final_quota_H)
        _apply_policy(sim_fork, country="F", tau_rate=final_tau_F, quota_mult=final_quota_F)
        sim_fork.run(lookahead)

        J_H = float(objective.compute(sim_fork.history, "H", "F"))
        J_F = float(objective.compute(sim_fork.history, "F", "H"))

        if config.opt_config.select == "sum":
            score = J_H + J_F
        elif config.opt_config.select == "min":
            score = min(J_H, J_F)
        elif config.opt_config.select == "H":
            score = J_H
        elif config.opt_config.select == "F":
            score = J_F
        else:
            raise ValueError(f"Unknown select: {config.opt_config.select}")

        if best_score is None or score > best_score:
            best_score = score
            best = (final_tau_H, final_tau_F, final_quota_H, final_quota_F)
            best_J = (J_H, J_F)

    tau_H_vec, tau_F_vec, quota_H_vec, quota_F_vec = best

    policies = {
        "H": {"tariff": _policy_dict_from_vector(tau_H_vec, config.constraints.active_sectors),
              "quota": _policy_dict_from_vector(quota_H_vec, config.constraints.active_sectors)},
        "F": {"tariff": _policy_dict_from_vector(tau_F_vec, config.constraints.active_sectors),
              "quota": _policy_dict_from_vector(quota_F_vec, config.constraints.active_sectors)},
    }
    info = {"J_H_pred": float(best_J[0]), "J_F_pred": float(best_J[1]), "samples": samples}
    return policies, info


def run_grad_experiment(config: GameConfig) -> TwoCountryDynamicSimulator:
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
        policies, pred = _optimize_simultaneous_search(sim, config=config, prev_tau_H=prev_tau_H, prev_tau_F=prev_tau_F)
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
    print("策略决策摘要")
    print("=" * 80)
    for rec in policy_history:
        print(f"\n轮次 {rec['round']}:")
        print(f"  决策(H): {rec['decision']['H']}")
        print(f"  决策(F): {rec['decision']['F']}")
        print(f"  预测效用: {rec['predicted']}")
        print(f"  事后效用: {rec['payoff']}")
        print(f"  搜索耗时: {rec['opt_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  搜索总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮搜索: {sum(timing_stats) / len(timing_stats):.2f}s")
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
    # 例子：决策间隔 10 期、lookahead 8 期、热启动 1000 期；
    # trigger 在热启动后发生；目标函数为 standard（只关注本国效益）。
    cfg = GameConfig(
        name="search_simultaneous_trigger",
        # NOTE (experiment): 这里打开按供给归一的价格调整公式（归一化后的公式 16）。
        normalize_gap_by_supply=True,
        # NOTE (experiment): 启用归一化后，原先的 theta_price=12500 会导致价格指数爆炸（exp 溢出）。
        # 需要重新标定；这里给一个温和的起点，便于先跑通实验流程。
        theta_price=0.1,
        rounds=10,
        decision_interval=10,
        lookahead_periods=15,
        warmup_periods=1000,
        trigger_settle_periods=0,
        trigger=TriggerConfig(country="H", tariff={4: 0.5}),
        opt_config=OptimizationConfig(
            iterations=200,
            multi_start=8,
            start_strategy="noisy_current",
            select="sum",
            seed=42,
        ),
        constraints=ConstraintsConfig(active_sectors=[2, 3], reciprocal_coeff=0, max_tariff=1.0, min_quota=0.0),
        objective=ObjectiveConfig(type="standard", weights=(1.0, 1.0, 1.0)),
        plot=True,
        output_dir="results_grad_search",
    )
    run_grad_experiment(cfg)
