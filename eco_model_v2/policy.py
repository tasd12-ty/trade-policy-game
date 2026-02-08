"""政策事件与时间线。

定义政策动作（关税、配额、TFP 冲击等）和事件调度器。

依赖：types.py, simulator.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np


# ---- 政策动作 ----

@dataclass(frozen=True)
class TariffAction:
    """关税变更动作。"""
    country: str             # "H" 或 "F"
    rates: Dict[int, float]  # {部门: 关税率}


@dataclass(frozen=True)
class QuotaAction:
    """出口配额动作。"""
    country: str                    # "H" 或 "F"
    multipliers: Dict[int, float]   # {部门: 配额乘子}


@dataclass(frozen=True)
class TFPShock:
    """全要素生产率冲击。"""
    country: str             # "H" 或 "F"
    shocks: Dict[int, float] # {部门: TFP 乘子}


PolicyAction = Union[TariffAction, QuotaAction, TFPShock]


# ---- 事件调度 ----

@dataclass
class PolicyEvent:
    """定时策略事件。"""
    period: int             # 触发时期
    action: PolicyAction    # 策略动作
    label: str = ""         # 可选描述


@dataclass
class PolicyTimeline:
    """策略事件时间线。

    管理一组按时期排序的策略事件。
    仿真器在每期开始时检查是否有待触发的事件。
    """
    events: List[PolicyEvent] = field(default_factory=list)

    def add(self, period: int, action: PolicyAction, label: str = "") -> None:
        """添加事件。"""
        self.events.append(PolicyEvent(period=period, action=action, label=label))
        self.events.sort(key=lambda e: e.period)

    def get_events_at(self, period: int) -> List[PolicyEvent]:
        """获取指定时期的所有事件。"""
        return [e for e in self.events if e.period == period]

    def apply_to_simulator(self, sim, period: int) -> List[str]:
        """应用指定时期的事件到仿真器。

        返回已应用事件的描述列表。
        """
        applied = []
        for event in self.get_events_at(period):
            action = event.action

            if isinstance(action, TariffAction):
                sim.apply_tariff(action.country, action.rates)
                applied.append(
                    f"t={period}: {action.country} 关税变更 {action.rates}"
                )

            elif isinstance(action, QuotaAction):
                sim.apply_quota(action.country, action.multipliers)
                applied.append(
                    f"t={period}: {action.country} 配额变更 {action.multipliers}"
                )

            elif isinstance(action, TFPShock):
                _apply_tfp_shock(sim, action)
                applied.append(
                    f"t={period}: {action.country} TFP 冲击 {action.shocks}"
                )

            if event.label:
                applied[-1] += f" [{event.label}]"

        return applied


def _apply_tfp_shock(sim, action: TFPShock) -> None:
    """应用 TFP 冲击（修改 A 参数）。"""
    import dataclasses
    from .types import TwoCountryParams

    if action.country == "H":
        cp = sim.params.home
    else:
        cp = sim.params.foreign

    new_A = np.array(cp.A, copy=True, dtype=float)
    for sector, mult in action.shocks.items():
        if 0 <= sector < cp.Nl:
            new_A[sector] *= max(float(mult), 0.01)

    new_cp = dataclasses.replace(cp, A=new_A)

    if action.country == "H":
        sim._params = TwoCountryParams(home=new_cp, foreign=sim.params.foreign)
    else:
        sim._params = TwoCountryParams(home=sim.params.home, foreign=new_cp)
    sim._rebuild_engines()
