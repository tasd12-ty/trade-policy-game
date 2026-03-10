"""dynamics.py 测试 — eq 19-26。"""

import numpy as np
import pytest

from eco_model_v2.types import CountryParams, CountryState
from eco_model_v2.dynamics import CountryDynamics
from eco_model_v2.production import compute_output
from eco_model_v2.factors import compute_income


@pytest.fixture
def dynamics_setup(small_params):
    """构造初始状态和引擎。"""
    cp = small_params
    Nl, M = cp.Nl, cp.M_factors
    price = np.ones(Nl + M)
    X_dom = np.ones((Nl, Nl + M)) * 0.5
    X_imp = np.ones((Nl, Nl)) * 0.2
    X_imp[:, :cp.Ml] = 0.0
    C_dom = np.ones(Nl) * 0.5
    C_imp = np.ones(Nl) * 0.2
    C_imp[:cp.Ml] = 0.0
    imp_price = cp.import_cost * 1.0
    export_base = cp.exports.copy()
    export_actual = cp.exports.copy()
    output_prod = compute_output(cp.A, cp.alpha, X_dom, X_imp, cp.gamma, cp.rho, cp.Ml, M)
    output = np.concatenate([output_prod, cp.L.copy()])
    income = compute_income(price, cp.L, Nl)

    state = CountryState(
        X_dom=X_dom, X_imp=X_imp, C_dom=C_dom, C_imp=C_imp,
        price=price, imp_price=imp_price,
        export_base=export_base, export_actual=export_actual,
        output=output, income=float(income),
    )
    engine = CountryDynamics(cp, tau=0.2)
    return engine, state, imp_price


class TestCountryDynamics:
    def test_step_returns_valid_state(self, dynamics_setup):
        engine, state, imp_price = dynamics_setup
        new_state = engine.step(state, imp_price)
        assert new_state.price.shape == state.price.shape
        assert new_state.output.shape == state.output.shape
        assert np.isfinite(new_state.price).all()
        assert np.isfinite(new_state.output).all()
        assert (new_state.price > 0).all()

    def test_10_steps_no_nan(self, dynamics_setup):
        engine, state, imp_price = dynamics_setup
        s = state
        for _ in range(10):
            s = engine.step(s, imp_price)
        assert not np.isnan(s.price).any()
        assert not np.isnan(s.output).any()
        assert not np.isnan(s.X_dom).any()
        assert not np.isnan(s.X_imp).any()

    def test_all_positive(self, dynamics_setup):
        engine, state, imp_price = dynamics_setup
        s = state
        for _ in range(5):
            s = engine.step(s, imp_price)
        assert (s.price > 0).all()
        assert (s.X_dom >= 0).all()
        assert (s.X_imp >= 0).all()
        assert (s.C_dom >= 0).all()
        assert (s.C_imp >= 0).all()

    def test_income_update_eq26(self, dynamics_setup):
        """eq 26: I_{t+1} 由 t 期要素价格和 t+1 期要素使用量决定。"""
        engine, state, imp_price = dynamics_setup
        new_state = engine.step(state, imp_price)

        Nl = engine.Nl
        M = engine.M
        factor_usage = new_state.X_dom[:, Nl:Nl + M].sum(axis=0)
        expected_income = float(np.dot(state.price[Nl:Nl + M], factor_usage))
        assert new_state.income == pytest.approx(expected_income, rel=1e-10, abs=1e-10)
