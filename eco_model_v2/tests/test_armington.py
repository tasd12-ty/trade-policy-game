"""armington.py 测试 — eq 4 CES, eq 9/15 θ。"""

import numpy as np
import pytest

from eco_model_v2.armington import (
    armington_quantity,
    armington_price,
    armington_share_from_prices,
    theta_from_quantities,
)


class TestArmingtonQuantity:
    """eq 4 中的 CES 聚合: [γ·x_d^ρ + (1-γ)·x_f^ρ]^{α/ρ}。"""

    def test_equal_inputs(self):
        """x_d == x_f, gamma=0.5, alpha=1 → 结果 = x_d。"""
        q = armington_quantity(0.5, 1.0, 1.0, alpha=1.0, rho=0.5)
        assert abs(q - 1.0) < 1e-6

    def test_alpha_one(self):
        """α=1, γ=1 → 纯国内品 x_d^1。"""
        q = armington_quantity(1.0, 2.0, 5.0, alpha=1.0, rho=0.5)
        assert abs(q - 2.0) < 1e-4

    def test_alpha_zero(self):
        """α=0 → 不使用该投入，返回 1。"""
        q = armington_quantity(0.5, 2.0, 3.0, alpha=0.0, rho=0.5)
        assert abs(q - 1.0) < 1e-6

    def test_rho_zero_cobb_douglas(self):
        """ρ→0 → Cobb-Douglas: exp(α·(γ·ln(x_d) + (1-γ)·ln(x_f)))。"""
        gamma, alpha = 0.6, 0.8
        x_d, x_f = 2.0, 3.0
        q = armington_quantity(gamma, x_d, x_f, alpha=alpha, rho=1e-10)
        expected = np.exp(alpha * (gamma * np.log(x_d) + (1 - gamma) * np.log(x_f)))
        assert abs(q - expected) < 1e-4


class TestArmingtonPrice:
    """CES 对偶价格: [γ^σ · p_d^{1-σ} + (1-γ)^σ · p_f^{1-σ}]^{1/(1-σ)}。"""

    def test_equal_prices_rho_zero(self):
        """ρ→0 (σ→1), p_d == p_f → P* = p / (γ^γ·(1-γ)^{1-γ})。"""
        gamma = 0.5
        p = armington_price(gamma, 1.0, 1.0, rho=1e-10)
        # CD dual includes entropy: P* = 1/(0.5^0.5 * 0.5^0.5) = 2.0
        expected = 1.0 / (gamma ** gamma * (1 - gamma) ** (1 - gamma))
        assert abs(p - expected) < 1e-3

    def test_positive(self):
        """CES 价格应为正。"""
        p = armington_price(0.5, 2.0, 3.0, rho=0.5)
        assert p > 0

    def test_rho_zero(self):
        """ρ→0 (σ→1) → CD dual: P_d^γ·P_f^{1-γ} / (γ^γ·(1-γ)^{1-γ})。"""
        gamma = 0.7
        p_d, p_f = 2.0, 3.0
        p = armington_price(gamma, p_d, p_f, rho=1e-10)
        geom = np.exp(gamma * np.log(p_d) + (1 - gamma) * np.log(p_f))
        entropy = gamma ** gamma * (1 - gamma) ** (1 - gamma)
        expected = geom / entropy
        assert abs(p - expected) < 1e-3


class TestThetaFromQuantities:
    """eq 9/15 用量端 θ。"""

    def test_shape(self):
        gamma = np.array([[1.0, 0.5], [1.0, 0.5]])
        x_d = np.ones((2, 2))
        x_f = np.ones((2, 2))
        rho = np.array([[0.0, 0.3], [0.0, 0.3]])
        theta = theta_from_quantities(gamma, x_d, x_f, rho)
        assert theta.shape == (2, 2)

    def test_equal_quantities(self):
        """x_d == x_f, γ=0.5 → θ=0.5。"""
        gamma = np.array([[0.5]])
        x_d = np.array([[2.0]])
        x_f = np.array([[2.0]])
        rho = np.array([[0.3]])
        theta = theta_from_quantities(gamma, x_d, x_f, rho)
        assert abs(theta[0, 0] - 0.5) < 1e-4

    def test_rho_zero_gives_gamma(self):
        """ρ→0 → θ = γ。"""
        gamma = np.array([[0.6]])
        x_d = np.array([[2.0]])
        x_f = np.array([[3.0]])
        rho = np.array([[1e-10]])
        theta = theta_from_quantities(gamma, x_d, x_f, rho)
        assert abs(theta[0, 0] - 0.6) < 1e-4

    def test_gamma_one(self):
        """γ=1 → θ=1（纯国内品）。"""
        gamma = np.array([[1.0]])
        theta = theta_from_quantities(gamma, np.array([[1.0]]), np.array([[5.0]]), np.array([[0.5]]))
        assert abs(theta[0, 0] - 1.0) < 1e-6
