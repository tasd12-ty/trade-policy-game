"""生产、成本与收入计算。"""

from __future__ import annotations

import numpy as np

from .armington import armington_price, armington_quantity
from .types import CountryParams
from .utils import EPS, safe_log


def value_added_share(params: CountryParams) -> np.ndarray:
    """增加值份额 v_i = 1 - sum_j alpha_ij。"""
    row_sum = np.asarray(params.alpha, dtype=float).sum(axis=1)
    return np.clip(1.0 - row_sum, 1e-8, None)


def compute_output(params: CountryParams, X_dom: np.ndarray, X_imp: np.ndarray, tradable_mask: np.ndarray) -> np.ndarray:
    """按生产函数计算各部门产出。"""
    n = params.n
    out = np.zeros(n, dtype=float)
    for i in range(n):
        prod = max(float(params.A[i]), EPS)
        for j in range(n):
            a = float(params.alpha[i, j])
            if a <= 0.0:
                continue
            if bool(tradable_mask[j]):
                qty = armington_quantity(
                    gamma=float(params.gamma[i, j]),
                    x_dom=float(X_dom[i, j]),
                    x_for=float(X_imp[i, j]),
                    alpha=a,
                    rho=float(params.rho[i, j]),
                )
                prod *= max(qty, EPS)
            else:
                prod *= max(float(X_dom[i, j]), EPS) ** a
        out[i] = max(prod, EPS)
    return out


def compute_marginal_cost(params: CountryParams, prices: np.ndarray, import_prices: np.ndarray, tradable_mask: np.ndarray) -> np.ndarray:
    """按对偶成本函数计算边际成本 lambda_i。"""
    n = params.n
    lambdas = np.zeros(n, dtype=float)
    for i in range(n):
        log_cost = -float(safe_log(params.A[i]))
        for j in range(n):
            a = float(params.alpha[i, j])
            if a <= 0.0:
                continue
            if bool(tradable_mask[j]):
                p_star = float(
                    armington_price(
                        gamma=float(params.gamma[i, j]),
                        p_dom=float(prices[j]),
                        p_for=float(import_prices[j]),
                        rho=float(params.rho[i, j]),
                    )
                )
                log_cost += a * float(safe_log(p_star))
            else:
                log_cost += a * float(safe_log(prices[j]))
        lambdas[i] = float(np.exp(log_cost))
    return np.clip(lambdas, EPS, None)


def compute_income(params: CountryParams, prices: np.ndarray, outputs: np.ndarray) -> float:
    """收入近似：I = sum_i P_i * Y_i * v_i。"""
    va = value_added_share(params)
    return float(np.sum(np.asarray(prices) * np.asarray(outputs) * va))
