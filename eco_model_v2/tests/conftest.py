"""测试公用 fixtures。"""

import numpy as np
import pytest

from eco_model_v2.types import CountryParams, TwoCountryParams


@pytest.fixture
def small_params():
    """2 部门 (1 non-tradable + 1 tradable), 1 factor 的参数。"""
    Nl, Ml, M = 2, 1, 1
    alpha = np.array([[0.3, 0.3, 0.4],
                      [0.2, 0.4, 0.4]])
    gamma = np.array([[1.0, 0.5],
                      [1.0, 0.5]])
    rho = np.array([[0.0, 0.3],
                    [0.0, 0.3]])
    beta = np.array([0.4, 0.6])
    gamma_cons = np.array([1.0, 0.5])
    rho_cons = np.array([0.0, 0.3])
    A = np.array([1.0, 1.0])
    L = np.array([10.0])
    import_cost = np.array([1.1, 1.1])
    exports = np.array([0.5, 0.5, 0.0])

    return CountryParams(
        alpha=alpha, gamma=gamma, rho=rho,
        beta=beta, A=A, exports=exports,
        gamma_cons=gamma_cons, rho_cons=rho_cons,
        import_cost=import_cost, L=L,
        Ml=Ml, M_factors=M,
    )


@pytest.fixture
def two_country_params(small_params):
    """对称两国参数。"""
    import dataclasses
    foreign = dataclasses.replace(
        small_params,
        alpha=small_params.alpha.copy(),
        gamma=small_params.gamma.copy(),
        rho=small_params.rho.copy(),
        beta=small_params.beta.copy(),
        A=small_params.A.copy(),
        exports=small_params.exports.copy(),
        gamma_cons=small_params.gamma_cons.copy(),
        rho_cons=small_params.rho_cons.copy(),
        import_cost=small_params.import_cost.copy(),
        L=small_params.L.copy(),
    )
    return TwoCountryParams(home=small_params, foreign=foreign)
