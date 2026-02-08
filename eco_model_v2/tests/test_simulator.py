"""simulator.py 测试。"""

import numpy as np
import pytest

from eco_model_v2.presets import make_symmetric_params
from eco_model_v2.simulator import TwoCountrySimulator


@pytest.fixture
def sim():
    params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
    s = TwoCountrySimulator(params, tau=0.2)
    s.initialize()
    return s


class TestTwoCountrySimulator:
    def test_initialize_and_run(self, sim):
        sim.run(10)
        assert sim.t == 11  # initial + 10 steps
        assert sim.current_state["H"] is not None
        assert sim.current_state["F"] is not None

    def test_no_nan_after_run(self, sim):
        sim.run(20)
        h = sim.current_state["H"]
        assert np.isfinite(h.price).all()
        assert np.isfinite(h.output).all()

    def test_apply_tariff(self, sim):
        sim.run(5)
        old_ic = sim.params.home.import_cost.copy()
        sim.apply_tariff("H", {2: 0.3})
        new_ic = sim.params.home.import_cost
        assert new_ic[2] > old_ic[2]

    def test_fork_isolation(self, sim):
        sim.run(10)
        forked = sim.fork()
        forked.apply_tariff("H", {2: 0.5})
        forked.run(5)
        # Original should not be affected
        assert sim.t == 11
        np.testing.assert_array_equal(
            sim.params.home.import_cost,
            make_symmetric_params(Nl=3, Ml=1, M_factors=1).home.import_cost,
        )

    def test_observation(self, sim):
        sim.run(5)
        obs = sim.get_observation("H")
        assert "income" in obs
        assert "price" in obs
        assert "trade_balance" in obs
        assert obs["country"] == "H"

    def test_payoff_computation(self, sim):
        sim.run(20)
        payoff = sim.compute_payoff("H", start_period=0)
        assert isinstance(payoff, float)
        assert np.isfinite(payoff)

    def test_supply_chain_no_history_mutation(self):
        """供应链覆盖层不应修改历史状态的 export_base。"""
        from eco_model_v2.supply_chain import (
            SupplyChainNetwork, SupplyChainNode, disruption_transform,
        )

        params = make_symmetric_params(Nl=3, Ml=1, M_factors=1)
        net = SupplyChainNetwork()
        net.add_edge(
            SupplyChainNode("H", 1),
            SupplyChainNode("F", 1),
            disruption_transform(severity=0.5),
        )

        s = TwoCountrySimulator(params, tau=0.2, supply_chain=net)
        s.initialize()

        # 记录初始 export_base
        h0_export = s.history["H"][0].export_base.copy()
        s.run(5)

        # 初始状态的 export_base 不应被修改
        np.testing.assert_array_equal(
            s.history["H"][0].export_base, h0_export,
            "供应链覆盖层不应修改历史状态",
        )
