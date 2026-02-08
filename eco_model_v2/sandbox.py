"""经济沙盒：策略博弈框架。

参考 grad_op/analysis/optimization/grad_game.py 设计。
支持多种策略类型（固定、梯度、LLM）的博弈环境。

依赖：simulator.py, policy.py, presets.py, types.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from .types import TwoCountryParams
from .simulator import TwoCountrySimulator
from .supply_chain import SupplyChainNetwork
from .utils import EPS


# ---- 博弈配置 ----

@dataclass(frozen=True)
class GameConfig:
    """博弈实验配置。"""
    name: str = "default"
    rounds: int = 10                   # 博弈轮数
    decision_interval: int = 20        # 每轮决策间隔期数
    warmup_periods: int = 50           # 预热期数
    lookahead_periods: int = 20        # 前瞻期数（用于策略评估）
    # 触发事件
    trigger_country: str | None = None     # 首先行动的国家
    trigger_tariff: Dict[int, float] = field(default_factory=dict)
    trigger_quota: Dict[int, float] = field(default_factory=dict)
    trigger_settle_periods: int = 10   # 触发后稳定期数
    # 策略类型
    players: Dict[str, str] = field(default_factory=lambda: {"H": "fixed", "F": "fixed"})
    # 约束
    active_sectors: List[int] = field(default_factory=list)
    max_tariff: float = 1.0
    min_quota: float = 0.0
    # 目标函数权重
    income_weight: float = 1.0
    trade_weight: float = 0.5
    price_stability_weight: float = 0.2


@dataclass
class RoundResult:
    """单轮博弈结果。"""
    round_num: int
    decisions: Dict[str, Dict[str, Any]]   # {"H": {"tariff": ..., "quota": ...}, "F": ...}
    payoffs: Dict[str, float]              # {"H": payoff, "F": payoff}
    observations: Dict[str, Dict]          # {"H": obs_dict, "F": obs_dict}
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult:
    """完整博弈结果。"""
    config: GameConfig
    rounds: List[RoundResult]
    total_payoffs: Dict[str, float]
    history_length: int


# ---- 经济沙盒 ----

class EconomicSandbox:
    """经济沙盒：支持多种策略的博弈环境。

    用法：
        config = GameConfig(rounds=10, decision_interval=20)
        sandbox = EconomicSandbox(params, config)
        result = sandbox.run_game(agents={"H": agent_h, "F": agent_f})
    """

    def __init__(
        self,
        params: TwoCountryParams,
        config: GameConfig,
        tau: float = 0.3,
        supply_chain: SupplyChainNetwork | None = None,
    ):
        self.config = config
        self.sim = TwoCountrySimulator(
            params, tau=tau, supply_chain=supply_chain,
        )

    def initialize(self) -> None:
        """初始化仿真环境。"""
        self.sim.initialize()

    def initialize_from_equilibrium(self) -> None:
        """从均衡态初始化。"""
        self.sim.initialize_from_equilibrium()

    def get_observation(self, country: str) -> Dict:
        """获取当前经济观测。"""
        return self.sim.get_observation(country)

    def apply_action(
        self,
        country: str,
        action: Dict[str, Any],
    ) -> None:
        """施加政策动作。

        参数：
            country: "H" 或 "F"
            action:  {"tariff": {sector: rate}, "quota": {sector: mult}}
        """
        tariff = action.get("tariff", {})
        quota = action.get("quota", {})

        if tariff:
            self.sim.apply_tariff(country, tariff)
        if quota:
            self.sim.apply_quota(country, quota)

    def step_environment(self, periods: int) -> None:
        """推进仿真。"""
        self.sim.run(periods)

    def compute_payoff(
        self,
        country: str,
        start_period: int = 0,
    ) -> float:
        """计算效用/收益。"""
        return self.sim.compute_payoff(
            country, start_period,
            income_weight=self.config.income_weight,
            trade_weight=self.config.trade_weight,
            price_stability_weight=self.config.price_stability_weight,
        )

    def fork(self) -> EconomicSandbox:
        """分叉沙盒，用于前瞻搜索。"""
        new_sandbox = EconomicSandbox.__new__(EconomicSandbox)
        new_sandbox.config = self.config
        new_sandbox.sim = self.sim.fork()
        return new_sandbox

    # ---- 博弈运行 ----

    def run_game(
        self,
        agents: Dict[str, Any],
    ) -> GameResult:
        """运行完整博弈。

        参数：
            agents: {"H": PolicyAgent, "F": PolicyAgent}
                    每个 agent 需实现 observe(context) 和 decide() 方法。

        返回：
            GameResult
        """
        config = self.config

        # 1. 初始化
        if not self.sim.history["H"]:
            self.initialize()

        # 2. 预热
        self.step_environment(config.warmup_periods)

        # 3. 触发事件
        if config.trigger_country:
            trigger_action = {}
            if config.trigger_tariff:
                trigger_action["tariff"] = config.trigger_tariff
            if config.trigger_quota:
                trigger_action["quota"] = config.trigger_quota
            if trigger_action:
                self.apply_action(config.trigger_country, trigger_action)
            self.step_environment(config.trigger_settle_periods)

        # 4. 主博弈循环
        round_results = []
        for r in range(config.rounds):
            start_period = self.sim.t

            # 4.1 收集观测并通知智能体
            observations = {}
            for c in ["H", "F"]:
                obs = self.get_observation(c)
                observations[c] = obs
                if hasattr(agents[c], "observe"):
                    agents[c].observe(obs)

            # 4.2 收集决策
            decisions = {}
            for c in ["H", "F"]:
                if hasattr(agents[c], "decide"):
                    decision = agents[c].decide()
                else:
                    decision = {"tariff": {}, "quota": {}}
                decisions[c] = decision

            # 4.3 同时施加动作
            for c in ["H", "F"]:
                self.apply_action(c, decisions[c])

            # 4.4 推进仿真
            self.step_environment(config.decision_interval)

            # 4.5 计算收益
            payoffs = {}
            for c in ["H", "F"]:
                payoffs[c] = self.compute_payoff(c, start_period)

            round_results.append(RoundResult(
                round_num=r + 1,
                decisions=decisions,
                payoffs=payoffs,
                observations=observations,
            ))

        # 5. 汇总
        total_payoffs = {"H": 0.0, "F": 0.0}
        for rr in round_results:
            for c in ["H", "F"]:
                total_payoffs[c] += rr.payoffs[c]

        return GameResult(
            config=config,
            rounds=round_results,
            total_payoffs=total_payoffs,
            history_length=self.sim.t,
        )
