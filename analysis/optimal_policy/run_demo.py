"""可配置演示：按部门优化出口管制与进口关税（SPSA/PGD 对比）。

功能要点
- 触发策略（trigger）在代码内写死配置，便于复现实验；不通过命令行输入。
- 目标：价格波动更小、收入更高、净贸易差更大，默认等权，时间平均（默认 10 期内）。
- 决策：选取前若干可贸易部门（默认 2 个），对每个部门优化出口乘子 m∈[0,1] 与关税率 τ∈[0,1]（可通过 --tau-max 调整上限）。
- 求解器：同时运行 SPSA 与 PGD，便于比较；实现解耦，后续可添加其他方法。

仅用于分析，不修改核心代码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# 兼容直接执行：将仓库根目录加入 sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.optimal_policy.objective import (
    build_baseline,
    evaluate_objective,
    decision_vector_for_sectors,
    apply_per_sector_from_x,
)
from analysis.optimal_policy.spsa_opt import spsa, SPSAConfig, pgd_fd

# -------------------------
# 触发策略（在此配置，避免命令行输入）
# -------------------------
# 示例：将出口乘子默认设为 1（无额外管制），关税设为 0（无额外关税）。
# 如需模拟“对手先动”，可填充对手国家的 trigger，例如：
# TRIGGER_FOREIGN_EXPORT = {2: 0.8}，TRIGGER_FOREIGN_TARIFF = {2: 0.1}
# 本脚本中 actor 决策会覆盖自身 trigger，同部门冲突时以后者为准。
TRIGGER_HOME_EXPORT: Dict[int, float] = {}
TRIGGER_HOME_TARIFF: Dict[int, float] = {0: 0.2, 1: 0.2, 2:0.2, 3:0.2, 4:0.2, 5:0.2}  # H 对部门0 征收 20% 关税
TRIGGER_FOREIGN_EXPORT: Dict[int, float] = {}
TRIGGER_FOREIGN_TARIFF: Dict[int, float] = {}


def _parse_weights(text: str) -> Tuple[float, float, float]:
    """解析权重字符串，默认 '1,1,1'。"""
    if not text:
        return (1.0, 1.0, 1.0)
    parts = [p for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("权重需形如 '1,1,1' 三个数字")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _fmt_vec(x: np.ndarray, sector_idxs: np.ndarray) -> str:
    K = len(sector_idxs)
    ms = ", ".join([f"m[{i}]={x[j]:.3f}" for j, i in enumerate(sector_idxs)])
    ts = ", ".join([f"tau[{i}]={x[K + j]:.3f}" for j, i in enumerate(sector_idxs)])
    return f"[{ms} | {ts}]"


def _print_result(label: str, x: np.ndarray, J: float, comps: dict, sector_idxs: np.ndarray) -> None:
    print(f"{label}: {_fmt_vec(x, sector_idxs)}")
    print(f"  J={J:+.4f} | comps={comps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="部门粒度政策最优演示（SPSA/PGD）")
    parser.add_argument("--horizon", type=int, default=10, help="评估周期（期数），含 t=0，建议 <=10")
    parser.add_argument("--max-sectors", type=int, default=2, help="参与优化的可贸易部门数量上限")
    parser.add_argument("--actor", type=str, default="H", choices=["H", "F"], help="优化对象国家")
    parser.add_argument("--aggregate", type=str, default="average", choices=["average", "terminal"], help="目标聚合方式")
    parser.add_argument("--weights", type=str, default="1,1,1", help="目标权重，形如 'w_price,w_income,w_trade'")
    parser.add_argument("--tau-max", type=float, default=1.0, help="关税率上限（例如 1.0 表示 +100%）")
    parser.add_argument("--burn-in", type=int, default=100, help="施策前的 burn-in 期数，让价格先贴稳")
    parser.add_argument("--price-scale", type=float, default=10000.0, help="价格分量放大倍数，用于提高读数权重")
    parser.add_argument("--methods", type=str, default="spsa,pgd", help="逗号分隔的求解器列表：spsa,pgd")
    parser.add_argument("--spsa-iters", type=int, default=60, help="SPSA 迭代步数")
    parser.add_argument("--spsa-a0", type=float, default=0.3, help="SPSA 初始步长 a0")
    parser.add_argument("--spsa-c0", type=float, default=0.1, help="SPSA 初始扰动 c0")
    parser.add_argument("--spsa-seed", type=int, default=42, help="SPSA 随机种子")
    parser.add_argument("--pgd-steps", type=int, default=40, help="PGD 步数")
    parser.add_argument("--pgd-step-size", type=float, default=0.15, help="PGD 步长")
    parser.add_argument("--pgd-h", type=float, default=1e-2, help="PGD 有限差分步长 h")
    parser.add_argument("--pgd-momentum", type=float, default=0.3, help="PGD 动量系数")
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    # 触发策略按国家区分，便于“对手先动、我方回应”
    fixed_exp = {"H": dict(TRIGGER_HOME_EXPORT), "F": dict(TRIGGER_FOREIGN_EXPORT)}
    fixed_tar = {"H": dict(TRIGGER_HOME_TARIFF), "F": dict(TRIGGER_FOREIGN_TARIFF)}

    if args.horizon < 0:
        raise ValueError("horizon 需为非负整数")
    if args.horizon > 50:
        print("提示：建议将 horizon 控制在 10 期以内，当前较大可能较慢。")

    baseline = build_baseline(horizon=args.horizon)
    x0, info = decision_vector_for_sectors(
        baseline.sim0,
        actor=args.actor,
        use_tradable_only=True,
        max_sectors=args.max_sectors,
    )
    sector_idxs = info["sector_idxs"]
    K = len(sector_idxs)
    if K == 0:
        raise RuntimeError("未找到可贸易部门，请检查模型参数。")

    lo = np.concatenate([np.zeros(K), np.zeros(K)])
    hi = np.concatenate([np.ones(K), np.full(K, float(args.tau_max))])

    def f(x: np.ndarray) -> Tuple[float, dict]:
        exp_map, tar_map = apply_per_sector_from_x(
            baseline,
            x,
            sector_idxs=sector_idxs,
            tau_max=args.tau_max,
        )
        # actor 决策覆盖自身 trigger，同部门取决策值
        actor_exp = {**fixed_exp.get(args.actor, {}), **exp_map}
        actor_tar = {**fixed_tar.get(args.actor, {}), **tar_map}
        J, comps = evaluate_objective(
            baseline,
            actor=args.actor,
            export_controls=actor_exp if actor_exp else None,
            import_tariffs=actor_tar if actor_tar else None,
            fixed_export_controls=fixed_exp,
            fixed_import_tariffs=fixed_tar,
            horizon=args.horizon,
            aggregate=args.aggregate,  # 默认 time-average
            weights=weights,
            burn_in=args.burn_in,
            price_scale=args.price_scale,
        )
        return float(J), comps

    # 基线（带 trigger，且 x0=无额外政策），用于对照
    J0, comps0 = f(x0)
    _print_result("基线（trigger + x0）", x0, J0, comps0, sector_idxs)
    print(f"触发策略 H: export={fixed_exp['H'] or '{}'}, tariff={fixed_tar['H'] or '{}'}")
    print(f"触发策略 F: export={fixed_exp['F'] or '{}'}, tariff={fixed_tar['F'] or '{}'}")
    print(f"参与优化的部门索引: {list(map(int, sector_idxs))}, tau_max={args.tau_max}, burn_in={args.burn_in}, price_scale={args.price_scale}")

    method_list = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    results: List[Tuple[str, np.ndarray, float, dict]] = []

    if "spsa" in method_list:
        x_spsa, val_spsa, info_spsa = spsa(
            x0,
            f,
            lo,
            hi,
            cfg=SPSAConfig(iterations=args.spsa_iters, a0=args.spsa_a0, c0=args.spsa_c0, seed=args.spsa_seed),
        )
        results.append(("SPSA", x_spsa, val_spsa, info_spsa))

    if "pgd" in method_list:
        x_pgd, val_pgd, info_pgd = pgd_fd(
            x0,
            f,
            lo,
            hi,
            steps=args.pgd_steps,
            h=args.pgd_h,
            step_size=args.pgd_step_size,
            momentum=args.pgd_momentum,
        )
        results.append(("PGD", x_pgd, val_pgd, info_pgd))

    for name, x_opt, J_opt, comps_opt in results:
        _print_result(f"{name} 最优", x_opt, J_opt, comps_opt, sector_idxs)

    if not results:
        print("未选择求解器（methods 为空），仅输出基线。")


if __name__ == "__main__":
    main()
