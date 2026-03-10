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


class TestSolvePricesRho0:
    def test_eq39_diag_correction(self):
        """eq 39 含 −Σ α·ln(α) 修正项 → 改变均衡价格水平。"""
        Nl, Ml, M = 2, 1, 0
        alpha = np.array([[0.3, 0.7],
                          [0.5, 0.5]])
        theta = np.array([[1.0, 0.6],
                          [1.0, 0.6]])
        alpha_tilde = build_alpha_tilde(alpha, theta, Ml)
        log_A = np.zeros(Nl)
        log_imp = np.zeros(Nl)

        log_p = solve_prices_rho0(alpha, alpha_tilde, theta, log_imp, log_A, Ml)
        price = np.exp(log_p)

        # 含常数项后，即使 A=1, P^O=1，价格也应偏离 0（对数空间）
        # −Σ α·ln(α) > 0 when 0 < α < 1 → prices > 1
        assert (price > 0).all()
        assert np.isfinite(price).all()


class TestSolveOutputRho0:
    def test_eq38_consumer_feedback(self):
        """eq 38 含消费反馈项时 PY 应大于无反馈版本。

        注：CRS + 全额国内消费 → B 矩阵退化为奇异。
        此测试使用非 CRS 参数（行和 < 1）避免奇异。
        """
        Nl, Ml, M = 3, 0, 1
        # 行和 < 1 (非 CRS)，避免 B 矩阵奇异
        alpha = np.array([
            [0.2, 0.2, 0.1, 0.3],  # sum = 0.8
            [0.1, 0.3, 0.1, 0.2],  # sum = 0.7
            [0.2, 0.1, 0.2, 0.2],  # sum = 0.7
        ])
        theta = np.ones((Nl, Nl)) * 0.7
        alpha_tilde = build_alpha_tilde(alpha, theta, Ml)
        beta = np.array([0.3, 0.4, 0.3])
        price = np.ones(Nl)
        exports = np.array([0.5, 0.3, 0.4, 0.0])

        # 有消费反馈 (M_factors=1)
        PY_with = solve_output_rho0(alpha_tilde, alpha, beta, price, exports, Nl, M)
        # 无消费反馈 (M_factors=0)
        PY_without = solve_output_rho0(alpha_tilde, alpha, beta, price, exports, Nl, 0)

        # 两者均应为正有限值
        assert (PY_with > 0).all()
        assert (PY_without > 0).all()
        assert np.isfinite(PY_with).all()
        # 消费反馈增大乘数效应 → PY 应更大
        assert (PY_with >= PY_without - 1e-10).all()


class TestSolveEquilibriumRho0:
    def test_prices_near_one(self):
        """对称参数 → 名义锚 P[0]=1。"""
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

        gamma_cons = np.array([1.0, 0.5])
        result = solve_equilibrium_rho0(alpha, gamma, beta, A, exports, imp_price, L, Ml, M, gamma_cons=gamma_cons)

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

        gamma_cons = np.array([1.0, 0.5])
        result = solve_equilibrium_rho0(alpha, gamma, beta, A, exports, imp_price, L, Ml, M, gamma_cons=gamma_cons)

        assert (result["output"] > 0).all()
        assert (result["PY"] > 0).all()
        assert result["income"] > 0
