"""eco_model_v2 —— 两国多部门 CGE 模型包。

基于 Project_1__LLM_Agents_as_Dynamic_Game_Players.pdf 全部公式实现，
支持要素市场、供应链扩展、策略博弈与 LLM 智能体接口。
"""

from .types import (
    CountryParams,
    TwoCountryParams,
    CountryBlock,
    StaticEquilibriumResult,
    CountryState,
)
from .production import compute_output, compute_cost, compute_marginal_cost
from .factors import (
    compute_income,
    compute_income_value_added,
    compute_factor_demand,
    factor_clearing_residual,
    factor_prices_rho0,
    income_rho0,
)
from .simulator import TwoCountrySimulator
from .sandbox import EconomicSandbox, GameConfig, GameResult
from .agent_interface import (
    PolicyAgent,
    FixedPolicyAgent,
    TitForTatAgent,
    LLMPolicyAgent,
)
from .presets import make_symmetric_params, apply_tariff, apply_quota
from .pipeline import run_simulation, run_game, run_llm_game
from .compat import from_io_params, from_io_params_two_country
from .gradient_agent import GradientPolicyAgent
from .plotting import summarize_history, plot_game_analysis
from . import model_defaults

__all__ = [
    # 数据结构
    "CountryParams",
    "TwoCountryParams",
    "CountryBlock",
    "StaticEquilibriumResult",
    "CountryState",
    # 计算函数
    "compute_output",
    "compute_cost",
    "compute_marginal_cost",
    "compute_income",
    "compute_income_value_added",
    "compute_factor_demand",
    "factor_clearing_residual",
    "factor_prices_rho0",
    "income_rho0",
    # 仿真与博弈
    "TwoCountrySimulator",
    "EconomicSandbox",
    "GameConfig",
    "GameResult",
    # 智能体
    "PolicyAgent",
    "FixedPolicyAgent",
    "TitForTatAgent",
    "LLMPolicyAgent",
    # 预设与工具
    "make_symmetric_params",
    "apply_tariff",
    "apply_quota",
    # 高层接口
    "run_simulation",
    "run_game",
    "run_llm_game",
    # 梯度优化
    "GradientPolicyAgent",
    # 可视化
    "summarize_history",
    "plot_game_analysis",
    # 数据桥接
    "from_io_params",
    "from_io_params_two_country",
    "model_defaults",
]
