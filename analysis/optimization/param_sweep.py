"""简单参数扫描：在固定快照上测试不同优化超参（SPSA/PGD），寻找更优回应。

用法示例：
    PYTHONPATH=$PYTHONPATH:/root/eco-model \
    python analysis/optimization/param_sweep.py --objective standard --actor H

输出：
    - 基线得分（无额外政策）
    - 不同超参组合下的最优目标值与对应策略向量
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from time import perf_counter

import numpy as np

# 兼容直接执行：将仓库根目录加入 sys.path，确保可导入 analysis 命名空间
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.model import (bootstrap_simulator, create_symmetric_parameters)
from analysis.optimization.objective import (
    decision_vector_for_sectors,
    apply_per_sector_from_x,
    evaluate_snapshot_objective,
)
from analysis.optimization.spsa_opt import spsa, SPSAConfig


def _build_env(burn_in: int = 5):
    params = create_symmetric_parameters()
    sim = bootstrap_simulator(params)
    if burn_in > 0:
        sim.run(burn_in)
    return sim


def _objective_fn(sim, actor: str, sector_idxs, tau_max: float, horizon: int, objective_type: str, weights):
    def f(x: np.ndarray) -> Tuple[float, dict]:
        p_exp, p_tar = apply_per_sector_from_x(None, x, sector_idxs=sector_idxs, tau_max=tau_max)
        score = evaluate_snapshot_objective(
            sim,
            actor=actor,
            export_controls=p_exp,
            import_tariffs=p_tar,
            horizon=horizon,
            objective_type=objective_type,
            weights=weights,
        )
        return float(score), {}
    return f


def _compute_metrics(sim, actor: str, export_controls, import_tariffs, horizon: int):
    """在当前快照上施策并跑 horizon 期，返回关键指标（平均值与末值）。"""
    run_sim = sim.fork(keep_history=False)
    if export_controls:
        run_sim.apply_export_control(actor, export_controls)
    if import_tariffs:
        run_sim.apply_import_tariff(actor, import_tariffs)
    if horizon > 0:
        run_sim.run(horizon)
    hist = run_sim.summarize_history()[actor]
    # 仅用 t=1..H 统计“施策后的”区间
    idx = slice(1, None)
    income_avg = float(np.mean(hist["income"][idx]))
    income_end = float(hist["income"][-1])
    tb_avg = float(np.mean(hist["trade_balance_val"][idx]))
    tb_end = float(hist["trade_balance_val"][-1])
    price_rel = hist["price_mean"][idx] / hist["price_mean"][0]
    price_mean_rel = float(np.mean(price_rel))
    price_std_rel = float(np.std(price_rel))
    return {
        "income_avg": income_avg,
        "income_end": income_end,
        "tb_avg": tb_avg,
        "tb_end": tb_end,
        "price_mean_rel": price_mean_rel,
        "price_std_rel": price_std_rel,
    }


def sweep(actor: str, objective_type: str, horizon: int, tau_max: float, weights):
    sim = _build_env()
    x0, info = decision_vector_for_sectors(sim, actor=actor, max_sectors=2)
    sector_idxs = info["sector_idxs"]
    K = len(sector_idxs)
    lo = np.concatenate([np.zeros(K), np.zeros(K)])
    hi = np.concatenate([np.ones(K), np.full(K, float(tau_max))])
    base_score = evaluate_snapshot_objective(
        sim, actor=actor, horizon=horizon, objective_type=objective_type, weights=weights
    )

    f = _objective_fn(sim, actor, sector_idxs, tau_max, horizon, objective_type, weights)

    results: List[Dict] = []

    # SPSA 扫描
    spsa_grids = [
        # {"iterations": 30, "a0": 0.05, "c0": 0.05, "seed": 1},
        # {"iterations": 50, "a0": 0.08, "c0": 0.05, "seed": 2},
        # {"iterations": 50, "a0": 0.12, "c0": 0.08, "seed": 3},
        # {"iterations": 80, "a0": 0.15, "c0": 0.08, "seed": 4},
        # {"iterations": 100, "a0": 0.20, "c0": 0.08, "seed": 11},
        # {"iterations": 120, "a0": 0.25, "c0": 0.10, "seed": 12},
        # {"iterations": 150, "a0": 0.30, "c0": 0.12, "seed": 13},
        # {"iterations": 200, "a0": 0.35, "c0": 0.15, "seed": 21},
        # {"iterations": 120, "a0": 0.15, "c0": 0.03, "seed": 31},
        # {"iterations": 500, "a0": 0.45, "c0": 0.18, "seed": 31},
        # 追加未测试的组合（更高迭代/不同扰动与步长）
        {"iterations": 250, "a0": 0.28, "c0": 0.07, "seed": 41},
        {"iterations": 300, "a0": 0.40, "c0": 0.12, "seed": 42},
        {"iterations": 350, "a0": 0.45, "c0": 0.15, "seed": 43},
        {"iterations": 250, "a0": 0.18, "c0": 0.04, "seed": 44},
        {"iterations": 300, "a0": 0.22, "c0": 0.06, "seed": 45},
    ]
    for cfg in spsa_grids:
        t0 = perf_counter()
        best_x, best_val, _ = spsa(
            np.clip(x0, lo, hi),
            f,
            lo,
            hi,
            cfg=SPSAConfig(
                iterations=cfg["iterations"],
                a0=cfg["a0"],
                c0=cfg["c0"],
                seed=cfg["seed"],
            ),
        )
        results.append({"method": "SPSA", **cfg, "score": best_val, "x": best_x, "time_sec": perf_counter() - t0})

    # 附加指标：为每个结果计算收入/贸易/价格指标
    enriched = []
    for r in results:
        x = r["x"]
        p_exp, p_tar = apply_per_sector_from_x(None, x, sector_idxs=sector_idxs, tau_max=tau_max)
        metrics = _compute_metrics(sim, actor, p_exp, p_tar, horizon)
        enriched.append({**r, "metrics": metrics})

    # 按得分排序
    enriched = sorted(enriched, key=lambda r: r["score"], reverse=True)
    return base_score, sector_idxs, enriched


def main():
    parser = argparse.ArgumentParser(description="参数扫描以寻找更优政策回应")
    parser.add_argument("--actor", type=str, default="H", choices=["H", "F"])
    parser.add_argument("--objective", type=str, default="standard", choices=["standard", "relative"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--tau-max", type=float, default=2.0)
    parser.add_argument(
        "--weights",
        type=str,
        default="1,1,1",
        help="形如 w_income,w_trade,w_price（收入、贸易差额、价格稳定性）",
    )
    args = parser.parse_args()

    w_parts = [float(p) for p in args.weights.split(",")]
    if len(w_parts) != 3:
        raise ValueError("weights 需为 3 个数，用逗号分隔")

    base_score, sectors, results = sweep(
        actor=args.actor,
        objective_type=args.objective,
        horizon=args.horizon,
        tau_max=args.tau_max,
        weights=tuple(w_parts),
    )

    print(f"Actor={args.actor}, objective={args.objective}, horizon={args.horizon}, tau_max={args.tau_max}")
    print(f"Sectors optimized: {list(map(int, sectors))}")
    print(f"Baseline (no extra policy) score: {base_score:.4f}")
    print("-" * 80)
    for r in results:
        x = r.pop("x")
        metrics = r.pop("metrics")
        time_sec = r.pop("time_sec", None)
        print(f"Method={r.pop('method')}, score={r.pop('score'):.4f}, params={r}")
        print(f"  x={np.array2string(x, precision=4, floatmode='fixed')}")
        print(
            f"  metrics: income_avg={metrics['income_avg']:.4f}, income_end={metrics['income_end']:.4f}, "
            f"tb_avg={metrics['tb_avg']:.4f}, tb_end={metrics['tb_end']:.4f}, "
            f"price_mean_rel={metrics['price_mean_rel']:.4f}, price_std_rel={metrics['price_std_rel']:.4f}"
        )
        if time_sec is not None:
            print(f"  time: {time_sec:.2f}s")
    print("-" * 80)


if __name__ == "__main__":
    main()
