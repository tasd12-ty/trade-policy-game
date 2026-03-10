"""compat.py from_io_params 桥接测试。"""

import os
import tempfile

import numpy as np
import pytest

from eco_model_v2.compat import from_io_params, from_io_params_two_country
from eco_model_v2.types import CountryParams, TwoCountryParams


def _write_csv_matrix(path, data, row_names, col_names):
    """写入带行/列索引的 CSV 矩阵（模拟 io_params 输出）。"""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + col_names)
        for i, row in enumerate(data):
            w.writerow([row_names[i]] + [f"{v:.6f}" for v in row])


def _write_csv_vector(path, data, index_names, col_name="value"):
    """写入带索引的 CSV 向量（模拟 io_params Series.to_frame() 输出）。"""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", col_name])
        for i, v in enumerate(data):
            w.writerow([index_names[i], f"{v:.6f}"])


@pytest.fixture
def io_params_dirs():
    """创建模拟 io_params 三类输出目录。"""
    Nl = 3
    sectors = ["D01", "D02", "D03"]

    with tempfile.TemporaryDirectory() as tmpdir:
        cat_a = os.path.join(tmpdir, "category_a")
        cat_b = os.path.join(tmpdir, "category_b")
        cat_c = os.path.join(tmpdir, "category_c")
        os.makedirs(cat_a)
        os.makedirs(cat_b)
        os.makedirs(cat_c)

        # Category A
        alpha_ij = np.array([
            [0.2, 0.3, 0.1],
            [0.1, 0.2, 0.2],
            [0.3, 0.1, 0.2],
        ])
        _write_csv_matrix(
            os.path.join(cat_a, "alpha_ij.csv"),
            alpha_ij, sectors, sectors,
        )
        beta = np.array([0.3, 0.4, 0.3])
        _write_csv_vector(os.path.join(cat_a, "beta_j.csv"), beta, sectors)
        theta_cj = np.array([0.8, 0.6, 0.9])
        _write_csv_vector(os.path.join(cat_a, "theta_cj.csv"), theta_cj, sectors)

        # Category B
        P_j_O = np.array([1.1, 1.2, 0.9])
        _write_csv_vector(os.path.join(cat_b, "P_j_O.csv"), P_j_O, sectors)
        E_j = np.array([0.5, 0.3, 0.8])
        _write_csv_vector(os.path.join(cat_b, "E_j.csv"), E_j, sectors)
        factor_params = np.array([1.0, 0.05, 100.0, 50.0])
        _write_csv_vector(
            os.path.join(cat_b, "factor_params.csv"),
            factor_params, ["w", "r", "K", "L"],
        )

        # Category C
        A_i = np.array([1.0, 1.2, 0.9])
        _write_csv_vector(os.path.join(cat_c, "A_i.csv"), A_i, sectors)
        gamma_ij = np.array([
            [0.7, 0.6, 0.8],
            [0.8, 0.7, 0.6],
            [0.6, 0.8, 0.7],
        ])
        _write_csv_matrix(
            os.path.join(cat_c, "gamma_ij.csv"),
            gamma_ij, sectors, sectors,
        )
        rho_cj = np.array([0.0, 0.0, 0.0])
        _write_csv_vector(os.path.join(cat_c, "rho_cj.csv"), rho_cj, sectors)

        yield cat_a, cat_b, cat_c, {
            "alpha_ij": alpha_ij,
            "beta": beta,
            "theta_cj": theta_cj,
            "P_j_O": P_j_O,
            "E_j": E_j,
            "A_i": A_i,
            "gamma_ij": gamma_ij,
            "rho_cj": rho_cj,
        }


class TestFromIoParams:
    def test_basic_construction(self, io_params_dirs):
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=2)

        assert isinstance(cp, CountryParams)
        assert cp.Nl == 3
        assert cp.M_factors == 2

    def test_alpha_shape(self, io_params_dirs):
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=2)

        # alpha: (Nl, Nl+M) = (3, 5)
        assert cp.alpha.shape == (3, 5)
        # 前 Nl 列 = io_params alpha_ij
        np.testing.assert_allclose(cp.alpha[:, :3], expected["alpha_ij"])

    def test_factor_alpha_residual(self, io_params_dirs):
        """要素列 = (1 − Σα) / M_factors。"""
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=2)

        for i in range(3):
            row_sum = expected["alpha_ij"][i].sum()
            remaining = max(1.0 - row_sum, 0.01)
            expected_factor = remaining / 2.0
            np.testing.assert_allclose(
                cp.alpha[i, 3], expected_factor, rtol=1e-6,
            )
            np.testing.assert_allclose(
                cp.alpha[i, 4], expected_factor, rtol=1e-6,
            )

    def test_beta_and_A(self, io_params_dirs):
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=1)

        np.testing.assert_allclose(cp.beta, expected["beta"])
        np.testing.assert_allclose(cp.A, expected["A_i"])

    def test_import_cost_and_exports(self, io_params_dirs):
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=1)

        np.testing.assert_allclose(cp.import_cost, expected["P_j_O"])
        np.testing.assert_allclose(cp.exports[:3], expected["E_j"])
        # 要素出口 = 0
        np.testing.assert_allclose(cp.exports[3:], 0.0)

    def test_gamma_and_rho(self, io_params_dirs):
        cat_a, cat_b, cat_c, expected = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=1)

        np.testing.assert_allclose(cp.gamma, expected["gamma_ij"])
        # rho 默认为 0 (Cobb-Douglas)
        np.testing.assert_allclose(cp.rho, 0.0)

    def test_factor_endowment(self, io_params_dirs):
        cat_a, cat_b, cat_c, _ = io_params_dirs
        cp = from_io_params(cat_a, cat_b, cat_c, M_factors=2)

        # L = [50.0, 100.0] (L, K from factor_params)
        assert cp.L.shape == (2,)
        assert cp.L[0] == 50.0  # L
        assert cp.L[1] == 100.0  # K


class TestFromIoParamsTwoCountry:
    def test_two_country(self, io_params_dirs):
        cat_a, cat_b, cat_c, _ = io_params_dirs
        params = from_io_params_two_country(
            home_dirs=(cat_a, cat_b, cat_c),
            foreign_dirs=(cat_a, cat_b, cat_c),
            M_factors=1,
        )

        assert isinstance(params, TwoCountryParams)
        assert params.home.Nl == 3
        assert params.foreign.Nl == 3
        # 两国参数相同（使用同一数据源）
        np.testing.assert_allclose(
            params.home.alpha, params.foreign.alpha,
        )
