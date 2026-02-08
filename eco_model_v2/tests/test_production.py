"""production.py 测试 — eq 4 生产函数, eq 5 成本, eq 10 边际成本。"""

import numpy as np
import pytest

from eco_model_v2.production import compute_output, compute_cost, compute_marginal_cost


@pytest.fixture
def prod_params():
    """2 部门, 1 factor。"""
    Nl, Ml, M = 2, 1, 1
    A = np.array([1.5, 2.0])
    alpha = np.array([[0.3, 0.3, 0.4],
                      [0.2, 0.4, 0.4]])
    gamma = np.array([[1.0, 0.5],
                      [1.0, 0.5]])
    rho = np.array([[0.0, 0.3],
                    [0.0, 0.3]])
    return A, alpha, gamma, rho, Ml, M


class TestComputeOutput:
    def test_positive(self, prod_params):
        A, alpha, gamma, rho, Ml, M = prod_params
        X_dom = np.ones((2, 3)) * 1.0
        X_imp = np.ones((2, 2)) * 0.5
        Y = compute_output(A, alpha, X_dom, X_imp, gamma, rho, Ml, M)
        assert Y.shape == (2,)
        assert (Y > 0).all()

    def test_zero_inputs(self, prod_params):
        A, alpha, gamma, rho, Ml, M = prod_params
        X_dom = np.zeros((2, 3))
        X_imp = np.zeros((2, 2))
        Y = compute_output(A, alpha, X_dom, X_imp, gamma, rho, Ml, M)
        assert (Y >= 0).all()

    def test_scale_with_A(self, prod_params):
        A, alpha, gamma, rho, Ml, M = prod_params
        X_dom = np.ones((2, 3))
        X_imp = np.ones((2, 2)) * 0.3
        Y1 = compute_output(A, alpha, X_dom, X_imp, gamma, rho, Ml, M)
        Y2 = compute_output(A * 2, alpha, X_dom, X_imp, gamma, rho, Ml, M)
        # Y ∝ A
        np.testing.assert_allclose(Y2, Y1 * 2, rtol=1e-6)


class TestComputeMarginalCost:
    def test_positive(self, prod_params):
        A, alpha, gamma, rho, Ml, M = prod_params
        price = np.array([1.0, 1.0, 0.5])
        imp_price = np.array([1.1, 1.1])
        lambdas = compute_marginal_cost(A, alpha, price, imp_price, gamma, rho, Ml, M)
        assert lambdas.shape == (2,)
        assert (lambdas > 0).all()

    def test_scale_with_prices(self, prod_params):
        A, alpha, gamma, rho, Ml, M = prod_params
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([2.0, 2.0, 2.0])
        imp1 = np.array([1.0, 1.0])
        imp2 = np.array([2.0, 2.0])
        mc1 = compute_marginal_cost(A, alpha, p1, imp1, gamma, rho, Ml, M)
        mc2 = compute_marginal_cost(A, alpha, p2, imp2, gamma, rho, Ml, M)
        # 边际成本随价格线性缩放（CD 特性）
        np.testing.assert_allclose(mc2, mc1 * 2, rtol=0.1)
