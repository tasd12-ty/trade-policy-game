"""政策最优化目标函数（仅用于分析，不修改核心模块）。

功能：
- 基于两国动态仿真器构造“无政策”的基线路径；
- 施加“出口管制（乘子）/进口关税（百分比）”两类政策；
- 计算组合目标：价格波动越小越好、收入越高越好、净贸易差（出口额-进口额）越大越好。

约定：
- 仅对单一国家进行优化（默认 'H'）；政策在整个评估期内保持常数；
- 出口管制乘子 ∈ [0,1]；关税率 ≥ 0；
- 聚合方式支持 'terminal'（末期值）与 'average'（时均）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional

import numpy as np

from analysis.model import (
    bootstrap_simulator,
    TwoCountryDynamicSimulator,
    SimulationConfig,
    create_symmetric_parameters,
)


Country = Literal["H", "F"]
AggMode = Literal["terminal", "average"]
PRICE_SCALE = 10_000.0  # 放大价格分量，便于读数


@dataclass
class Baseline:
    sim0: TwoCountryDynamicSimulator
    horizon: int
    metrics_H: Dict[str, np.ndarray]
    metrics_F: Dict[str, np.ndarray]


def build_baseline(horizon: int = 40) -> Baseline:
    """构造“无政策”基线仿真与指标序列。

    返回包含历史的仿真器，以及按国家划分的关键指标序列（见 sim.summarize_history）。
    """
    params = create_symmetric_parameters()
    sim = bootstrap_simulator(params, theta_price=0.05)
    # Record t=0 (equilibrium), then run horizon steps (history length = horizon+1)
    if horizon > 0:
        sim.run(horizon)
    hist = sim.summarize_history()
    return Baseline(sim0=sim, horizon=horizon, metrics_H=hist["H"], metrics_F=hist["F"])


def _safe_ratio(x: float, y: float, eps: float = 1e-9) -> float:
    return x / (y if abs(y) > eps else (eps if y >= 0 else -eps))


def _components_average(
    baseline_series: Dict[str, np.ndarray],
    target_series: Dict[str, np.ndarray],
) -> Tuple[float, float, float]:
    """按“时均”计算三个分量：
    - 价格波动性：z_price = - std(P_t / P0)，t 从 1 到 T（T 为最后期，跳过 t=0）
      说明：标准差越小越好，故前面取负号，且用 P0 归一化保证量纲无关。
    - 收入提升：z_income = mean_t (I_t / I0 - 1)
    - 净贸易差：z_trade = mean_t (TB_t) / denom，其中 denom=|TB_0| 若非零，否则取 t=0 的进口额做尺度。
    """
    P0 = float(baseline_series["price_mean"][0])
    I0 = float(baseline_series["income"][0])
    TB0 = float(baseline_series["trade_balance_val"][0])
    IM0 = float(baseline_series.get("import_value_val", baseline_series["import_value"])[0])

    # 构造评估区间：跳过 t=0（初始均衡），若只有一帧则退化为 {0}
    T = len(target_series["price_mean"]) - 1
    idxs = range(1, T + 1) if T >= 1 else range(0, 1)

    P_rel = np.array([_safe_ratio(float(target_series["price_mean"][t]), P0) for t in idxs], dtype=float)
    I_rel = np.array([_safe_ratio(float(target_series["income"][t]), I0) - 1.0 for t in idxs], dtype=float)
    TB_vals = np.array([float(target_series["trade_balance_val"][t]) for t in idxs], dtype=float)

    std_p = float(np.std(P_rel))
    mean_i = float(np.mean(I_rel))
    denom = abs(TB0) if abs(TB0) > 1e-9 else max(IM0, 1e-9)
    mean_tb = float(np.mean(TB_vals))
    z_price = -std_p
    z_income = mean_i
    z_trade = _safe_ratio(mean_tb, denom)
    return z_price, z_income, z_trade


def evaluate_objective(
    baseline: Baseline,
    *,
    actor: Country = "H",
    export_controls: Optional[Dict[int, float]] = None,
    import_tariffs: Optional[Dict[int, float]] = None,
    fixed_export_controls: Optional[Dict[Country, Dict[int, float]]] = None,
    fixed_import_tariffs: Optional[Dict[Country, Dict[int, float]]] = None,
    horizon: Optional[int] = None,
    aggregate: AggMode = "terminal",
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    burn_in: int = 0,
    price_scale: float = PRICE_SCALE,
) -> Tuple[float, Dict[str, float]]:
    """评估“常数政策”下的组合目标。

    返回 (J, comps)，其中 comps 包含三个分量：
    - z_income：收入相对提升（时均或末期）；
    - z_trade：净贸易差的相对尺度值（时均或末期）；
    - z_price：价格稳定性（负的价格指数波动，越大越好）。

    权重顺序统一为：(收入, 贸易差额, 价格稳定性)。
    J 为给定权重的加权平均，并将权重归一到总和为 3 的尺度，便于直观比较。
    """
    H = baseline.horizon if horizon is None else int(horizon)
    sim = baseline.sim0.fork(keep_history=False)
    # 先做 burn-in，让价格/产出接近动态稳态，再施加冲击
    if burn_in > 0:
        sim.run(int(burn_in))

    # 先施加“固定策略”（可用于对手的既定关税/配额或阶段性 trigger）
    if fixed_export_controls:
        for c, mapping in fixed_export_controls.items():
            if mapping:
                sim.apply_export_control(c, mapping, note="analysis: fixed export control")
    if fixed_import_tariffs:
        for c, mapping in fixed_import_tariffs.items():
            if mapping:
                sim.apply_import_tariff(c, mapping, note="analysis: fixed tariff")

    # 再施加当前 actor 的决策（若与固定策略重叠，将覆盖同部门）
    if export_controls:
        sim.apply_export_control(actor, export_controls, note="analysis: decision export control")
    if import_tariffs:
        sim.apply_import_tariff(actor, import_tariffs, note="analysis: decision tariff")

    if H > 0:
        sim.run(H)
    series = sim.summarize_history()[actor]
    base_series = baseline.metrics_H if actor == "H" else baseline.metrics_F

    if aggregate == "terminal":
        # 末期指标：价格采用“末期相对水平”，收入与贸易差同理（与“减少波动”的概念不同，仅作兼容）
        P0 = float(base_series["price_mean"][0])
        I0 = float(base_series["income"][0])
        TB0 = float(base_series["trade_balance_val"][0])
        IM0 = float(base_series.get("import_value_val", base_series["import_value"])[0])
        Pt = float(series["price_mean"][-1])
        It = float(series["income"][-1])
        TBt = float(series["trade_balance_val"][-1])
        zp = -(_safe_ratio(Pt, P0) - 1.0) * float(price_scale)
        zi = _safe_ratio(It, I0) - 1.0
        denom = abs(TB0) if abs(TB0) > 1e-9 else max(IM0, 1e-9)
        zt = _safe_ratio(TBt, denom)
    else:
        zp, zi, zt = _components_average(base_series, series)
        zp = zp * float(price_scale)

    w = np.array(weights, float)
    # 将权重归一到总和为 3，保持与其它目标实现的尺度一致
    w = 3.0 * (w / max(w.sum(), 1e-12))
    w_income, w_trade, w_price = w

    comps = {"z_price": float(zp), "z_income": float(zi), "z_trade": float(zt)}
    # 注意：zp 为“价格稳定性”项（负的波动），因此权重 w_price 应为正值
    J = float((w_income * zi + w_trade * zt + w_price * zp) / 3.0)
    return J, comps


def decision_vector_for_sectors(
    sim: TwoCountryDynamicSimulator,
    *,
    actor: Country = "H",
    use_tradable_only: bool = True,
    max_sectors: Optional[int] = 2,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """按“部门粒度”构造决策向量模板。

    - 选取可贸易部门（默认）中的前 max_sectors 个用于测试；
    - 决策向量 x 的结构为 [m_1..m_K, τ_1..τ_K]，长度 2K；
      其中 m_k∈[0,1] 为出口乘子，τ_k≥0 为关税率。
    返回 (x0, info)，x0 取 m=1、τ=0 的起点。
    """
    n = sim.home_state.price.numel()
    mask = sim.tradable_mask if use_tradable_only else np.ones(n, dtype=bool)
    all_idxs = np.arange(n, dtype=int)[mask]
    if max_sectors is not None and len(all_idxs) > max_sectors:
        idxs = all_idxs[: int(max_sectors)]
    else:
        idxs = all_idxs
    K = len(idxs)
    x0 = np.concatenate([np.ones(K, dtype=float), np.zeros(K, dtype=float)])
    info = {"sector_idxs": idxs}
    return x0, info


def apply_per_sector_from_x(
    base: Baseline,
    x: np.ndarray,
    *,
    sector_idxs: np.ndarray,
    tau_max: float = 1.0,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """将 x=[m_1..m_K, τ_1..τ_K] 映射为政策字典。

    - 前 K 个元素裁剪到 [0,1]，映射为出口乘子；
    - 后 K 个元素裁剪到 [0, τ_max]，映射为进口关税率；默认 τ_max=1。
    """
    x = np.asarray(x, dtype=float)
    idxs = np.asarray(sector_idxs, dtype=int)
    K = len(idxs)
    m = np.clip(x[:K], 0.0, 1.0)
    tau = np.clip(x[K:2 * K], 0.0, float(tau_max))
    exp_map = {int(i): float(m[j]) for j, i in enumerate(idxs)}
    tar_map = {int(i): float(tau[j]) for j, i in enumerate(idxs)}
    return exp_map, tar_map


__all__ = [
    "Baseline",
    "build_baseline",
    "evaluate_objective",
    "evaluate_snapshot_objective",
    "decision_vector_for_sectors",
    "apply_per_sector_from_x",
]


def evaluate_snapshot_objective(
    sim_snapshot: TwoCountryDynamicSimulator,
    *,
    actor: Country = "H",
    export_controls: Optional[Dict[int, float]] = None,
    import_tariffs: Optional[Dict[int, float]] = None,
    horizon: int = 10,
    objective_type: Literal["standard", "relative"] = "standard",
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    price_scale: float = PRICE_SCALE,
    verbose: bool = False,
) -> float:
    """基于当前仿真快照评估策略目标。

    与 evaluate_objective 不同，这里不从“无政策基线”开始，而是从给定的 sim_snapshot 开始（视为 t=0），
    向后推演 horizon 期。常用于博弈中的 Lookahead Optimization。

    参数:
        objective_type:
            - 'standard': 自身福利最大化 (w1*I + w2*TB + w3*Stab)
            - 'relative': 相对优势最大化 (w1*(I - I_opp) + w2*(TB - TB_opp) + w3*Stab)，注意价格维稳通常仍针对自身。

        weights:
            权重顺序为 (收入, 贸易差额, 价格稳定性)。其中价格稳定性为“负的价格指数标准差”，
            因此对应权重应为正值。
    """
    # 1. Fork 一份用于演化
    sim = sim_snapshot.fork(keep_history=False)
    
    # 2. 记录初始参照值 (t=0)
    # 注意：Sim 当前状态即为 t=0 状态，尚未 Apply 新政策
    # 但为了计算相对变化，我们需要记录当前的 Income, Price 等
    # Sim.metrics 计算是基于当前的 export/supply derived 的
    # 我们先不做 step，直接用当前的 auxiliary metrics
    # 但 simple way: 取最近一次 step 的结果，或者单纯认为 t=0 的值就是 snapshot 里的状态
    # 为稳妥，我们假设 snapshot 是刚 step 完的状态。
    # 我们需要 "base" values。
    # 由于 TwoCountryDynamicSimulator 没有直接存储 computable metrics 历史（只存于 history list 里的 state tensors）
    # 我们需要在 python 层面算一下当前 metric。
    # 这里为了简便，我们 step 0 次？不，我们需要 reference。
    # 简单的做法：Run 1 step without policy to look at metrics? No that changes state.
    # Correct way: use sim.compute_metrics(sim.home_state, ...) but sim API might not store it.
    # Workaround: Run 1 step and use it as base? Or use the first step of the lookahead as base?
    # Better: The objective definitions usually use "growth from t=0".
    # Let's run 1 step with NO new policy to get benchmarks?
    # Or just use the values from the first step of the actual run as the "base" (t=1 vs t=1)? No.
    # Let's look at `summarize_history`. It processes `sim.history`.
    # `sim.history` in a fresh fork contains just `[initial_state]`.
    # `summarize_history` can derive metrics from it.
    hist_0 = sim.summarize_history()
    
    def _get_start_val(c_stats, key_val, key_idx):
        # 优先用 _val 后缀的标量，否则取 mean
        if key_val in c_stats:
            return float(c_stats[key_val][-1])
        return float(c_stats[key_idx][-1])

    I0_self = _get_start_val(hist_0[actor], "income", "income")
    # P0 对价格稳定很重要
    P0_self = _get_start_val(hist_0[actor], "price_mean", "price_mean")
    # TB 用绝对值做分母
    TB0_self_val = _get_start_val(hist_0[actor], "trade_balance_val", "trade_balance_val")
    IM0_self_val = _get_start_val(hist_0[actor], "import_value_val", "import_value")
    denom_self = abs(TB0_self_val) if abs(TB0_self_val) > 1e-9 else max(IM0_self_val, 1e-9)

    opp = "F" if actor == "H" else "H"
    if objective_type == "relative":
        I0_opp = _get_start_val(hist_0[opp], "income", "income")
        TB0_opp_val = _get_start_val(hist_0[opp], "trade_balance_val", "trade_balance_val")
        IM0_opp_val = _get_start_val(hist_0[opp], "import_value_val", "import_value")
        denom_opp = abs(TB0_opp_val) if abs(TB0_opp_val) > 1e-9 else max(IM0_opp_val, 1e-9)

    # 3. 施加政策
    if export_controls:
        sim.apply_export_control(actor, export_controls)
    if import_tariffs:
        sim.apply_import_tariff(actor, import_tariffs)

    # 4. 运行 Horizon
    if horizon > 0:
        sim.run(horizon)
    
    # 5. 统计
    hist_T = sim.summarize_history()
    
    # Helper to calc score for a country
    def calc_score(c_code, I0, P0, denom):
        series = hist_T[c_code]
        # Skip t=0, use t=1..H
        # hist series length = 1 (init) + horizon
        # average relevant metrics
        # Income growth
        I_seq = series["income"][1:]
        I_hat = float(np.mean(I_seq / I0 - 1.0))
        
        # TB scaled
        TB_seq = series["trade_balance_val"][1:]
        TB_hat = float(np.mean(TB_seq)) / denom
        
        # Price stability (negative std of P/P0)
        P_seq = series["price_mean"][1:]
        P_rel = P_seq / P0
        sigma_P = -float(np.std(P_rel)) * price_scale
        
        return I_hat, TB_hat, sigma_P

    i_self, tb_self, stab_self = calc_score(actor, I0_self, P0_self, denom_self)
    
    w = np.array(weights, dtype=float)
    # Norm weights to sum to 3 for scale consistency
    w = 3.0 * (w / max(w.sum(), 1e-12))

    if objective_type == "standard":
        # J = w1*I + w2*TB + w3*Stab
        val = w[0] * i_self + w[1] * tb_self + w[2] * stab_self
        return float(val / 3.0)
    
    elif objective_type == "relative":
        # J = w1*(I - I_opp) + w2*(TB - TB_opp) + w3*Stab_self
        # Note: We usually don't care about opp's price stability, or we might want to destabilize them?
        # The prompt def says: price term is self-focused.
        i_opp, tb_opp, _ = calc_score(opp, I0_opp, 1.0, denom_opp) # P0 not needed for I/TB
        
        # Relative terms
        rel_i = i_self - i_opp
        # For TB: TB_self is surplus. TB_opp is usually -TB_self in 2-country closed system?
        # In this model, TB_H + TB_F = 0 theoretically if conservation holds.
        # So TB_self - TB_opp = TB_self - (-TB_self) = 2 * TB_self.
        # So relative TB is just maximizing self TB really, but let's follow formula.
        rel_tb = tb_self - tb_opp 
        
        val = w[0] * rel_i + w[1] * rel_tb + w[2] * stab_self
        return float(val / 3.0)
    
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
