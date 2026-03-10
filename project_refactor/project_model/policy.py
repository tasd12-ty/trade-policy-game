"""策略事件与时间线执行。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pipeline import bootstrap_simulator
from .presets import create_symmetric_params
from .types import TwoCountryParams


@dataclass(frozen=True)
class PolicyEvent:
    """标准化政策事件。"""

    kind: str
    actor: str
    sectors: Dict[int, float]
    start_period: int
    end_period: Optional[int] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class ConflictBlock:
    """冲突语法糖：自动展开为事件列表。"""

    export_controls: Dict[str, Dict[int, float]] = field(default_factory=dict)
    import_tariffs: Dict[str, Dict[int, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    """时间线仿真配置。"""

    total_periods: int
    conflict_start: int
    tau_price: float = 0.1
    normalize_gap_by_supply: bool = False
    solver_max_iter: int = 400
    solver_tol: float = 1e-8
    events: Optional[List[PolicyEvent]] = None
    conflict: Optional[ConflictBlock] = None


def _events_from_conflict(block: ConflictBlock, start: int) -> List[PolicyEvent]:
    out: List[PolicyEvent] = []
    for actor, mapping in (block.export_controls or {}).items():
        out.append(PolicyEvent(kind="export_quota", actor=actor, sectors=dict(mapping), start_period=int(start)))
    for actor, mapping in (block.import_tariffs or {}).items():
        out.append(PolicyEvent(kind="import_tariff", actor=actor, sectors=dict(mapping), start_period=int(start)))
    return out


def run_simulation(config: SimulationConfig, params: Optional[TwoCountryParams] = None):
    """按事件时间线运行仿真。"""
    if params is None:
        params = create_symmetric_params()

    sim = bootstrap_simulator(
        params,
        tau_price=float(config.tau_price),
        normalize_gap_by_supply=bool(config.normalize_gap_by_supply),
        max_iterations=int(config.solver_max_iter),
        tolerance=float(config.solver_tol),
    )

    timeline: List[PolicyEvent] = []
    if config.conflict is not None:
        timeline.extend(_events_from_conflict(config.conflict, config.conflict_start))
    if config.events is not None:
        timeline.extend(list(config.events))

    start_bucket: Dict[int, List[PolicyEvent]] = {}
    end_bucket: Dict[int, List[PolicyEvent]] = {}
    for ev in timeline:
        start_bucket.setdefault(int(ev.start_period), []).append(ev)
        if ev.end_period is not None:
            end_bucket.setdefault(int(ev.end_period), []).append(ev)

    for t in range(int(config.total_periods)):
        if t in end_bucket:
            for ev in end_bucket[t]:
                if ev.kind == "export_quota":
                    sim.reset_export_control(ev.actor, sectors=list(ev.sectors.keys()), note=f"结束期 {t} 还原出口政策")
                elif ev.kind in ("import_tariff", "import_multiplier"):
                    sim.reset_import_policies(ev.actor, sectors=list(ev.sectors.keys()), note=f"结束期 {t} 还原进口政策")

        if t in start_bucket:
            for ev in start_bucket[t]:
                if ev.kind == "export_quota":
                    sim.apply_export_control(ev.actor, dict(ev.sectors), note=ev.note)
                elif ev.kind == "import_tariff":
                    sim.apply_import_tariff(ev.actor, dict(ev.sectors), note=ev.note)
                elif ev.kind == "import_multiplier":
                    sim.set_import_multiplier(ev.actor, dict(ev.sectors), relative_to_baseline=True, note=ev.note)

        sim.step()

    return sim
