"""factors.py 测试 — eq 7 收入, eq 8 要素需求, eq 29 出清, eq 40-41。"""

import numpy as np
import pytest

from eco_model_v2.factors import (
    compute_income,
    compute_factor_demand,
    factor_clearing_residual,
    factor_prices_rho0,
    income_rho0,
)


class TestComputeIncome:
    def test_basic(self):
        """I = Σ P_{Nl+k} · L_k。"""
        price = np.array([1.0, 1.5, 2.0, 0.8])  # Nl=2, M=2
        L = np.array([10.0, 5.0])
        income = compute_income(price, L, Nl=2)
        expected = 2.0 * 10.0 + 0.8 * 5.0
        assert abs(income - expected) < 1e-10

    def test_single_factor(self):
        price = np.array([1.0, 1.0, 0.5])  # Nl=2, M=1
        L = np.array([10.0])
        income = compute_income(price, L, Nl=2)
        assert abs(income - 5.0) < 1e-10


class TestFactorClearingResidual:
    def test_cleared_market(self):
        """当需求 = 供给时残差应为零。"""
        Nl, M = 2, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        price = np.array([1.0, 1.0, 0.5])
        L = np.array([10.0])
        # 构造 output 使得 Σ α_{i,Nl+k} · P_i · Y_i = P_{Nl+k} · L_k
        # α_{0,2}·P_0·Y_0 + α_{1,2}·P_1·Y_1 = P_2·L_0
        # 0.4·1·Y_0 + 0.4·1·Y_1 = 0.5·10 = 5
        # → Y_0 + Y_1 = 12.5
        Y = np.array([6.25, 6.25])
        residual = factor_clearing_residual(alpha, price, Y, L, Nl, M)
        assert abs(residual[0]) < 1e-10


class TestFactorPricesRho0:
    def test_basic(self):
        """P_{Nl+k} = (Σ_i α_{i,Nl+k} · PY_i) / L_k。"""
        Nl, M = 2, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        PY = np.array([5.0, 3.0])
        L = np.array([10.0])
        fp = factor_prices_rho0(alpha, PY, L, Nl, M)
        expected = (0.4 * 5.0 + 0.4 * 3.0) / 10.0
        assert abs(fp[0] - expected) < 1e-10


class TestIncomeRho0:
    def test_basic(self):
        """I = Σ_k (Σ_i α_{i,Nl+k} · PY_i)。"""
        Nl, M = 2, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        PY = np.array([5.0, 3.0])
        income = income_rho0(alpha, PY, Nl, M)
        expected = 0.4 * 5.0 + 0.4 * 3.0
        assert abs(income - expected) < 1e-10
