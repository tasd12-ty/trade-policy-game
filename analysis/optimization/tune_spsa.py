"""SPSA 超参搜索（交互式两国博弈）

目的：
- 自动测试不同 SPSA 超参数组合，找出在给定场景（冲击 + 多轮博弈）下表现更好的配置。
- 输出每组配置的得分与关键指标，并保存 CSV 便于后续分析。

说明：
- 本脚本只调“优化器超参”（SPSA 相关 + 部分决策维度参数），不改经济模型结构。
- 评价指标使用与 lookahead 目标一致的三项：收入增长、贸易差额、价格稳定性（负标准差）。
- 所有注释为中文；图表如需扩展请保持英文。
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# 兼容直接执行：将仓库根目录加入 sys.path，确保可导入 analysis 命名空间
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.model import bootstrap_simulator, create_symmetric_parameters
from analysis.optimization.interaction import InteractiveGame, GameConfig


def _score_from_history(
    series: Dict[str, np.ndarray],
    *,
    weights: Tuple[float, float, float],
    price_scale: float,
) -> Tuple[float, Dict[str, float]]:
    """从 summarize_history 的序列计算单国综合得分。

    权重顺序：(收入, 贸易差额, 价格稳定性)
    价格稳定性定义为：-std(price_mean / price0) * price_scale
    """
    w_income, w_trade, w_price = map(float, weights)
    w_sum = max(w_income + w_trade + w_price, 1e-12)
    # 归一到总和为 3，便于不同设置可比
    w_income, w_trade, w_price = [3.0 * w / w_sum for w in (w_income, w_trade, w_price)]

    if len(series["income"]) <= 1:
        return 0.0, {"z_income": 0.0, "z_trade": 0.0, "z_price": 0.0}

    I0 = float(series["income"][0])
    P0 = float(series["price_mean"][0])
    TB0 = float(series["trade_balance_val"][0])
    IM0 = float(series.get("import_value_val", series.get("import_value", np.array([0.0])))[0])
    denom = abs(TB0) if abs(TB0) > 1e-9 else max(IM0, 1e-9)

    idx = slice(1, None)
    inc = series["income"][idx]
    tb = series["trade_balance_val"][idx]
    pr = series["price_mean"][idx]

    z_income = float(np.mean(inc / max(I0, 1e-9) - 1.0))
    z_trade = float(np.mean(tb) / denom)
    z_price = -float(np.std(pr / max(P0, 1e-9))) * float(price_scale)

    J = float((w_income * z_income + w_trade * z_trade + w_price * z_price) / 3.0)
    return J, {"z_income": z_income, "z_trade": z_trade, "z_price": z_price}


def _run_one(
    cfg: GameConfig,
    *,
    rounds: int,
    warmup_periods: int,
    shock: Dict[str, Any],
    hide_warmup: bool,
) -> Dict[str, Any]:
    """运行一组配置并返回评估结果。"""
    params = create_symmetric_parameters()
    sim = bootstrap_simulator(params)

    if warmup_periods > 0:
        sim.run(int(warmup_periods))

    # 施加外生冲击
    c = shock.get("country", "H")
    if shock.get("tariff"):
        sim.apply_import_tariff(c, shock["tariff"], note=shock.get("note", "Shock"))
    if shock.get("export"):
        sim.apply_export_control(c, shock["export"], note=shock.get("note", "Shock"))

    # 若隐藏热启动，则以“冲击后状态”为新的 t=0
    if hide_warmup:
        h0 = sim.home_state.detach()
        f0 = sim.foreign_state.detach()
        sim.history = {"H": [h0], "F": [f0]}
        sim.policy_events = []

    game = InteractiveGame(cfg, initial_sim=sim)

    t0 = perf_counter()
    for _ in range(int(rounds)):
        game.step()
    elapsed = perf_counter() - t0

    hist = game.sim.summarize_history()
    J_H, comps_H = _score_from_history(hist["H"], weights=cfg.weights, price_scale=cfg.price_scale)
    J_F, comps_F = _score_from_history(hist["F"], weights=cfg.weights, price_scale=cfg.price_scale)

    # 这里默认用“两国得分均值”作为搜索目标；你也可以改成只看 H 或只看 F
    J_mean = 0.5 * (J_H + J_F)
    out = {
        "score_mean": J_mean,
        "score_H": J_H,
        "score_F": J_F,
        "elapsed_s": elapsed,
        "comps_H": comps_H,
        "comps_F": comps_F,
    }
    return out


def _candidate_grid() -> List[Dict[str, Any]]:
    """给一组“覆盖面更广”的候选超参组合（可按需要扩充）。"""
    iters = [30, 50, 80, 120]
    a0s = [0.03, 0.05, 0.08, 0.12, 0.18]
    c0s = [0.04, 0.07, 0.10, 0.15]
    restarts = [1, 2]
    # 经典衰减（可扩展）
    alphas = [0.602]
    gammas = [0.101]
    # 多起点会显著增加耗时，默认不纳入大网格；如需开启，可在 base_cfg 里单独指定
    # multi_starts = [1, 2, 4]

    out: List[Dict[str, Any]] = []
    for it in iters:
        for a0 in a0s:
            for c0 in c0s:
                for r in restarts:
                    for aa in alphas:
                        for gg in gammas:
                            out.append(
                                {
                                    "opt_iter": it,
                                    "opt_lr": a0,
                                    "opt_perturb": c0,
                                    "spsa_restarts": r,
                                    "spsa_alpha": aa,
                                    "spsa_gamma": gg,
                                }
                            )
    return out


def main() -> None:
    # 固定场景参数：避免每次手改，先用保守默认；你可按实验需求调整
    rounds = 5
    interval = 10
    warmup_periods = 50
    hide_warmup = True

    # 外生冲击：让策略优化有“回应对象”
    shock = {
        "country": "H",
        "tariff": {2: 0.2, 3: 0.2},
        "export": None,
        "note": "Initial shock: H tariff +20% on sectors 2&3",
    }

    # 固定不搜索的“结构性参数”（可在这里调整）
    base_cfg = GameConfig(
        objective_type_H="standard",
        objective_type_F="standard",
        strategy_mode_H="independent",
        strategy_mode_F="independent",
        reciprocal_alpha=1.0,
        opt_horizon=10,
        step_interval=interval,
        # 权重顺序：(收入, 贸易差额, 价格稳定性)
        weights=(1.0, 1.0, 1.0),
        # 降低价格项放大倍数，避免“过度维稳”
        price_scale=500.0,
        # 决策维度：优化更多部门通常更“有戏”，但更慢
        max_sectors=4,
        tau_max=2.0,
        # SPSA 默认值会在候选组合中覆盖
        spsa_seed=42,
    )

    candidates = _candidate_grid()
    print(f"Total candidates: {len(candidates)}")

    records: List[Dict[str, Any]] = []
    for i, hp in enumerate(candidates, start=1):
        cfg = GameConfig(**{**asdict(base_cfg), **hp})
        try:
            res = _run_one(cfg, rounds=rounds, warmup_periods=warmup_periods, shock=shock, hide_warmup=hide_warmup)
        except Exception as e:
            records.append(
                {
                    **hp,
                    "ok": False,
                    "error": repr(e),
                }
            )
            continue

        records.append(
            {
                **hp,
                "ok": True,
                "score_mean": res["score_mean"],
                "score_H": res["score_H"],
                "score_F": res["score_F"],
                "elapsed_s": res["elapsed_s"],
                "H_z_income": res["comps_H"]["z_income"],
                "H_z_trade": res["comps_H"]["z_trade"],
                "H_z_price": res["comps_H"]["z_price"],
                "F_z_income": res["comps_F"]["z_income"],
                "F_z_trade": res["comps_F"]["z_trade"],
                "F_z_price": res["comps_F"]["z_price"],
            }
        )

        if i % 10 == 0:
            print(f"Progress: {i}/{len(candidates)}")

    df = pd.DataFrame(records)
    ok = df[df["ok"] == True].copy()
    ok = ok.sort_values(["score_mean", "score_H"], ascending=False)

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "tune_spsa_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    if not ok.empty:
        top = ok.head(10)
        print("\nTop-10 (by score_mean):")
        cols = ["score_mean", "score_H", "score_F", "elapsed_s", "opt_iter", "opt_lr", "opt_perturb", "spsa_restarts", "spsa_alpha", "spsa_gamma"]
        print(top[cols].to_string(index=False))
    else:
        print("No successful runs. Check errors in CSV.")


if __name__ == "__main__":
    main()
