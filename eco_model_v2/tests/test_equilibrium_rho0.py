"""equilibrium_rho0.py 测试 — eq 37-41。"""

import numpy as np
import pytest

from eco_model_v2.equilibrium_rho0 import (
    build_alpha_tilde,
    solve_prices_rho0,
    solve_output_rho0,
    solve_equilibrium_rho0,
)


class TestBuildAlphaTilde:
    def test_shape(self):
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        theta = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        at = build_alpha_tilde(alpha, theta, Ml=1)
        assert at.shape == (2, 2)

    def test_non_tradable_unmodified(self):
        """非贸易品列 θ=1 → α̃ = α。"""
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        theta = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        at = build_alpha_tilde(alpha, theta, Ml=1)
        np.testing.assert_allclose(at[:, 0], alpha[:, 0])

    def test_tradable_modified(self):
        """可贸易品列不受 θ 影响（eq 37 定义）。"""
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        theta = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        at = build_alpha_tilde(alpha, theta, Ml=1)
        # 可贸易列直接复制 alpha
        np.testing.assert_allclose(at[:, 1], alpha[:, 1])


class TestSolveEquilibriumRho0:
    def test_prices_near_one(self):
        """对称参数 → 价格应接近 1。"""
        Nl, Ml, M = 2, 1, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        gamma = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        beta = np.array([0.4, 0.6])
        A = np.ones(Nl)
        exports = np.array([0.5, 0.5, 0.0])
        imp_price = np.array([1.0, 1.0])
        L = np.array([10.0])

        result = solve_equilibrium_rho0(alpha, gamma, beta, A, exports, imp_price, L, Ml, M)

        # P[0] = 1 (nominal anchor)
        assert abs(result["price"][0] - 1.0) < 1e-6
        # 价格应为正有限值
        assert (result["price"] > 0).all()
        assert np.isfinite(result["price"]).all()

    def test_positive_output(self):
        Nl, Ml, M = 2, 1, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        gamma = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        beta = np.array([0.4, 0.6])
        A = np.ones(Nl)
        exports = np.array([0.5, 0.5, 0.0])
        imp_price = np.array([1.1, 1.1])
        L = np.array([10.0])

        result = solve_equilibrium_rho0(alpha, gamma, beta, A, exports, imp_price, L, Ml, M)

        assert (result["output"] > 0).all()
        assert (result["PY"] > 0).all()
        assert result["income"] > 0
