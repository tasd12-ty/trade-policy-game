"""重构版模型最小示例。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_model import (  # noqa: E402
    ConflictBlock,
    PolicyEvent,
    SimulationConfig,
    bootstrap_simulator,
    create_symmetric_params,
    run_simulation,
)


def demo_bootstrap() -> None:
    """演示：静态均衡 + 手动施策 + 动态运行。"""
    params = create_symmetric_params(n=6)
    sim = bootstrap_simulator(params, tau_price=0.08, normalize_gap_by_supply=True)

    # 初始冲击：H 国对部门 2 加征 20% 进口关税
    sim.apply_import_tariff("H", {2: 0.2}, note="demo tariff")
    sim.run(20)

    summary = sim.summarize_history()
    print("[Bootstrap Demo] H 最终收入:", float(summary["H"]["income"][-1]))
    print("[Bootstrap Demo] F 最终收入:", float(summary["F"]["income"][-1]))


def demo_timeline() -> None:
    """演示：时间线事件驱动。"""
    cfg = SimulationConfig(
        total_periods=30,
        conflict_start=5,
        tau_price=0.08,
        normalize_gap_by_supply=True,
        conflict=ConflictBlock(import_tariffs={"F": {4: 0.3}}),
        events=[
            PolicyEvent(kind="export_quota", actor="H", sectors={3: 0.6}, start_period=10, end_period=20, note="阶段性出口管制"),
        ],
    )
    sim = run_simulation(cfg, params=create_symmetric_params(n=6))
    summary = sim.summarize_history()
    print("[Timeline Demo] H 最终贸易差额:", float(summary["H"]["trade_balance"][-1]))
    print("[Timeline Demo] F 最终贸易差额:", float(summary["F"]["trade_balance"][-1]))


if __name__ == "__main__":
    demo_bootstrap()
    demo_timeline()
