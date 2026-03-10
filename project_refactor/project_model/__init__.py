"""Project_1 重构版模型包。

目标：
1. 以模块化方式实现论文核心公式；
2. 保持静态均衡与动态仿真链路可复用；
3. 通过中文注释提升可维护性与可审计性。
"""

from .types import (
    CountryParams,
    TwoCountryParams,
    CountryBlock,
    StaticEquilibriumResult,
    CountryState,
)
from .presets import create_symmetric_params, from_raw_dict
from .equilibrium import solve_static_equilibrium
from .dynamics import TwoCountrySimulator
from .pipeline import bootstrap_simulator
from .policy import ConflictBlock, PolicyEvent, SimulationConfig, run_simulation

__all__ = [
    "CountryParams",
    "TwoCountryParams",
    "CountryBlock",
    "StaticEquilibriumResult",
    "CountryState",
    "create_symmetric_params",
    "from_raw_dict",
    "solve_static_equilibrium",
    "TwoCountrySimulator",
    "bootstrap_simulator",
    "ConflictBlock",
    "PolicyEvent",
    "SimulationConfig",
    "run_simulation",
]
