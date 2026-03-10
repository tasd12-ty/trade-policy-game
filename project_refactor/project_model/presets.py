"""参数构造与兼容转换。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import CountryParams, TwoCountryParams


def _to_country_params(block: Dict[str, Any]) -> CountryParams:
    alpha = np.asarray(block["alpha_ij"], dtype=float)
    gamma = np.asarray(block["gamma_ij"], dtype=float)
    rho = np.asarray(block["rho_ij"], dtype=float)
    beta = np.asarray(block["beta_j"], dtype=float)
    A = np.asarray(block["A_i"], dtype=float)
    exports = np.asarray(block.get("Export_i", np.zeros_like(beta)), dtype=float)
    gamma_cons = np.asarray(block.get("gamma_cj", gamma[0]), dtype=float)
    rho_cons = np.asarray(block.get("rho_cj", rho[0]), dtype=float)
    import_cost = np.asarray(block.get("import_cost", np.ones_like(beta)), dtype=float)
    return CountryParams(
        alpha=alpha,
        gamma=gamma,
        rho=rho,
        beta=beta,
        A=A,
        exports=exports,
        gamma_cons=gamma_cons,
        rho_cons=rho_cons,
        import_cost=import_cost,
    )


def from_raw_dict(raw: Dict[str, Any]) -> TwoCountryParams:
    """将旧格式参数字典转换为新结构。"""
    home = _to_country_params(raw["H"])
    foreign = _to_country_params(raw["F"])
    tradable = np.asarray(sorted(raw["tradable_sectors"]), dtype=int)
    return TwoCountryParams(home=home, foreign=foreign, tradable_idx=tradable)


def create_symmetric_params(n: int = 6) -> TwoCountryParams:
    """生成与 grad_op 兼容的对称基线参数。"""
    alpha_base = 0.15
    alpha_h = np.full((n, n), alpha_base, dtype=float)
    alpha_f = np.full((n, n), alpha_base, dtype=float)
    np.fill_diagonal(alpha_h, 0.0)
    np.fill_diagonal(alpha_f, 0.0)

    tradable = [i for i in range(n) if i >= max(0, n - 4)]
    gamma_h = np.full((n, n), 0.5, dtype=float)
    gamma_f = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if (j not in tradable) or (i == j):
                gamma_h[i, j] = 1.0
                gamma_f[i, j] = 1.0

    rho_h = np.full((n, n), 0.2, dtype=float)
    rho_f = np.full((n, n), 0.2, dtype=float)
    beta_h = np.full(n, 1.0 / n, dtype=float)
    beta_f = np.full(n, 1.0 / n, dtype=float)
    A_h = np.ones(n, dtype=float)
    A_f = np.ones(n, dtype=float)
    export_h = np.zeros(n, dtype=float)
    export_f = np.zeros(n, dtype=float)
    gamma_c_h = gamma_h[0].copy()
    gamma_c_f = gamma_f[0].copy()
    rho_c_h = rho_h[0].copy()
    rho_c_f = rho_f[0].copy()
    import_cost = np.ones(n, dtype=float)

    home = CountryParams(
        alpha=alpha_h,
        gamma=gamma_h,
        rho=rho_h,
        beta=beta_h,
        A=A_h,
        exports=export_h,
        gamma_cons=gamma_c_h,
        rho_cons=rho_c_h,
        import_cost=import_cost,
    )
    foreign = CountryParams(
        alpha=alpha_f,
        gamma=gamma_f,
        rho=rho_f,
        beta=beta_f,
        A=A_f,
        exports=export_f,
        gamma_cons=gamma_c_f,
        rho_cons=rho_c_f,
        import_cost=import_cost,
    )
    return TwoCountryParams(home=home, foreign=foreign, tradable_idx=np.asarray(tradable, dtype=int))
