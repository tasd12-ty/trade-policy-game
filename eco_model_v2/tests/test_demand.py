"""demand.py 测试 — eq 11-18 动态需求, eq 31-36 静态需求。"""

import numpy as np
import pytest

from eco_model_v2.demand import (
    compute_delta_x_dom,
    compute_delta_x_imp,
    static_intermediate_demand_dom,
    static_intermediate_demand_imp,
    static_consumption_demand_dom,
    static_consumption_demand_imp,
)


@pytest.fixture
def demand_setup():
    Nl, Ml, M = 2, 1, 1
    alpha = np.array([[0.3, 0.3, 0.4],
                      [0.2, 0.4, 0.4]])
    lambdas = np.array([1.0, 0.8])
    outputs = np.array([5.0, 3.0])
    theta = np.array([[1.0, 0.6],
                      [1.0, 0.6]])
    price = np.array([1.0, 1.2, 0.5])
    X_dom = np.ones((2, 3)) * 0.5
    return Nl, Ml, M, alpha, lambdas, outputs, theta, price, X_dom


class TestDeltaXDom:
    def test_shape(self, demand_setup):
        Nl, Ml, M, alpha, lambdas, outputs, theta, price, X_dom = demand_setup
        delta = compute_delta_x_dom(alpha, lambdas, outputs, theta, price, X_dom, Ml, M)
        assert delta.shape == (Nl, Nl + M)

    def test_finite(self, demand_setup):
        Nl, Ml, M, alpha, lambdas, outputs, theta, price, X_dom = demand_setup
        delta = compute_delta_x_dom(alpha, lambdas, outputs, theta, price, X_dom, Ml, M)
        assert np.isfinite(delta).all()


class TestStaticDemand:
    def test_budget_consistency(self):
        """静态需求：总支出 = 收入。"""
        Nl, Ml, M = 2, 1, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        theta = np.array([[1.0, 0.6],
                          [1.0, 0.6]])
        beta = np.array([0.4, 0.6])
        theta_c = np.array([1.0, 0.6])
        price = np.array([1.0, 1.2, 0.5])
        imp_price = np.array([1.1, 1.3])
        output = np.array([5.0, 3.0])
        income = 5.0

        X_dom = static_intermediate_demand_dom(alpha, theta, price, output, Ml, M)
        X_imp = static_intermediate_demand_imp(alpha[:, :Nl], theta, imp_price, price, output, Ml)
        C_dom = static_consumption_demand_dom(beta, theta_c, income, price, Ml)
        C_imp = static_consumption_demand_imp(beta, theta_c, income, imp_price, Ml)

        # 消费支出
        cons_spending = float(
            np.dot(price[:Nl], C_dom) + np.dot(imp_price, C_imp)
        )
        assert abs(cons_spending - income) < income * 0.01, f"Budget violation: {cons_spending} vs {income}"
