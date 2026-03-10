"""sandbox.py 测试。"""

import numpy as np
import pytest

from eco_model_v2.presets import make_symmetric_params
from eco_model_v2.sandbox import EconomicSandbox, GameConfig
from eco_model_v2.agent_interface import FixedPolicyAgent, TitForTatAgent


class TestEconomicSandbox:
    def test_run_game_fixed_agents(self):
        """固定策略博弈完整运行。"""
        params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
        config = GameConfig(
            rounds=3,
            decision_interval=5,
            warmup_periods=10,
        )
        sandbox = EconomicSandbox(params, config, tau=0.2)
        sandbox.initialize()

        agents = {
            "H": FixedPolicyAgent(tariff={2: 0.1}),
            "F": FixedPolicyAgent(),
        }
        result = sandbox.run_game(agents)

        assert len(result.rounds) == 3
        assert "H" in result.total_payoffs
        assert "F" in result.total_payoffs
        assert result.history_length > 0

    def test_fork(self):
        """分叉沙盒独立运行。"""
        params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
        config = GameConfig(rounds=2, decision_interval=5, warmup_periods=5)
        sandbox = EconomicSandbox(params, config, tau=0.2)
        sandbox.initialize()
        sandbox.step_environment(10)

        original_t = sandbox.sim.t
        forked = sandbox.fork()
        forked.apply_action("H", {"tariff": {2: 0.5}})
        forked.step_environment(5)

        # 原始沙盒不受影响
        assert sandbox.sim.t == original_t
        # 分叉沙盒独立推进（fork 仅保留最后状态，t 从 1 开始）
        assert forked.sim.t == 1 + 5

    def test_trigger(self):
        """触发事件正常执行。"""
        params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
        config = GameConfig(
            rounds=2,
            decision_interval=5,
            warmup_periods=5,
            trigger_country="H",
            trigger_tariff={2: 0.3},
            trigger_settle_periods=5,
        )
        sandbox = EconomicSandbox(params, config, tau=0.2)
        sandbox.initialize()

        agents = {"H": FixedPolicyAgent(), "F": FixedPolicyAgent()}
        result = sandbox.run_game(agents)

        assert len(result.rounds) == 2
        assert result.history_length > 15  # warmup + settle + rounds

    def test_tit_for_tat_tracks_opponent_tariff(self):
        """Tit-for-Tat 应在下一轮复制对手关税水平。"""
        params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
        config = GameConfig(
            rounds=2,
            decision_interval=1,
            warmup_periods=0,
        )
        sandbox = EconomicSandbox(params, config, tau=0.2)
        sandbox.initialize()

        agents = {
            "H": FixedPolicyAgent(tariff={2: 0.3}),
            "F": TitForTatAgent(),
        }
        result = sandbox.run_game(agents)

        # 第 1 轮尚未观察到对手政策，第 2 轮应开始模仿
        f_round1 = result.rounds[0].decisions["F"]["tariff"]
        f_round2 = result.rounds[1].decisions["F"]["tariff"]
        assert f_round1 == {}
        assert f_round2.get(2, 0.0) == pytest.approx(0.3, rel=1e-2)
