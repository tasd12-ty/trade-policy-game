"""高层启动入口。

提供一站式接口：构造参数 → 初始化仿真 → 运行博弈 → 返回结果。

依赖：presets.py, simulator.py, sandbox.py, agent_interface.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .types import TwoCountryParams
from .presets import make_symmetric_params
from .simulator import TwoCountrySimulator
from .sandbox import EconomicSandbox, GameConfig, GameResult
from .agent_interface import (
    PolicyAgent,
    FixedPolicyAgent,
    TitForTatAgent,
    LLMPolicyAgent,
)
from .supply_chain import SupplyChainNetwork


def run_simulation(
    params: TwoCountryParams | None = None,
    periods: int = 200,
    tau: float = 0.3,
    from_equilibrium: bool = False,
    supply_chain: SupplyChainNetwork | None = None,
    **preset_kwargs,
) -> TwoCountrySimulator:
    """运行基础仿真（无博弈）。

    参数：
        params:          两国参数（提供则忽略 preset_kwargs）
        periods:         仿真期数
        tau:             价格调整速度
        from_equilibrium: 是否从均衡态启动
        supply_chain:    可选供应链网络
        **preset_kwargs: 传给 make_symmetric_params 的参数

    返回：
        TwoCountrySimulator（含历史记录）
    """
    if params is None:
        params = make_symmetric_params(**preset_kwargs)

    sim = TwoCountrySimulator(
        params, tau=tau, supply_chain=supply_chain,
    )

    if from_equilibrium:
        sim.initialize_from_equilibrium()
    else:
        sim.initialize()

    sim.run(periods)
    return sim


def run_game(
    params: TwoCountryParams | None = None,
    config: GameConfig | None = None,
    agents: Dict[str, PolicyAgent] | None = None,
    tau: float = 0.3,
    from_equilibrium: bool = False,
    supply_chain: SupplyChainNetwork | None = None,
    **preset_kwargs,
) -> GameResult:
    """运行策略博弈。

    参数：
        params:   两国参数（提供则忽略 preset_kwargs）
        config:   博弈配置（默认 10 轮）
        agents:   {"H": PolicyAgent, "F": PolicyAgent}
                  默认两国均为固定策略
        tau:      价格调整速度
        from_equilibrium: 是否从均衡态启动
        supply_chain: 可选供应链网络
        **preset_kwargs: 传给 make_symmetric_params 的参数

    返回：
        GameResult
    """
    if params is None:
        params = make_symmetric_params(**preset_kwargs)

    if config is None:
        config = GameConfig()

    if agents is None:
        agents = {
            "H": FixedPolicyAgent(),
            "F": FixedPolicyAgent(),
        }

    sandbox = EconomicSandbox(
        params, config, tau=tau, supply_chain=supply_chain,
    )

    if from_equilibrium:
        sandbox.initialize_from_equilibrium()
    else:
        sandbox.initialize()

    return sandbox.run_game(agents)


def run_llm_game(
    llm_client: Any,
    params: TwoCountryParams | None = None,
    config: GameConfig | None = None,
    llm_plays: str = "H",
    opponent_strategy: str = "fixed",
    opponent_tariff: Dict[int, float] | None = None,
    tau: float = 0.3,
    from_equilibrium: bool = False,
    **preset_kwargs,
) -> GameResult:
    """运行 LLM 博弈（便捷接口）。

    参数：
        llm_client:        LLM 客户端实例
        params:            两国参数
        config:            博弈配置
        llm_plays:         LLM 控制哪个国家 ("H", "F", "both")
        opponent_strategy: 对手策略 ("fixed", "tit_for_tat")
        opponent_tariff:   对手固定关税
        tau:               价格调整速度
        from_equilibrium:  是否从均衡态启动
        **preset_kwargs:   传给 make_symmetric_params 的参数

    返回：
        GameResult
    """
    if params is None:
        params = make_symmetric_params(**preset_kwargs)

    if config is None:
        config = GameConfig(
            players={"H": "llm" if "H" in llm_plays.upper() else "fixed",
                     "F": "llm" if "F" in llm_plays.upper() else "fixed"},
        )

    Ml = params.Ml
    Nl = params.Nl
    active = list(range(Ml, Nl))

    agents: Dict[str, PolicyAgent] = {}
    for c in ["H", "F"]:
        if llm_plays.upper() in (c, "BOTH"):
            agents[c] = LLMPolicyAgent(
                llm_client=llm_client,
                active_sectors=active,
                max_tariff=config.max_tariff,
                min_quota=config.min_quota,
            )
        elif opponent_strategy == "tit_for_tat":
            agents[c] = TitForTatAgent(initial_tariff=opponent_tariff or {})
        else:
            agents[c] = FixedPolicyAgent(tariff=opponent_tariff or {})

    return run_game(
        params=params, config=config, agents=agents,
        tau=tau, from_equilibrium=from_equilibrium,
    )
