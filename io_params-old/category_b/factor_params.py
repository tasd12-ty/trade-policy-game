"""B 类要素参数（w, r, K, L）计算。"""

from __future__ import annotations

import pandas as pd

from .external_inputs import ExternalBInputs
from .shared import EPS, safe_positive


def compute_w(external: ExternalBInputs) -> float:
    """计算工资参数 w。"""

    labor_population = safe_positive(external.labor_population, EPS)
    return float(external.labor_income_total) / labor_population


def compute_r(external: ExternalBInputs) -> float:
    """计算利率参数 r。"""

    return float(external.interest_rate_r)


def compute_k(external: ExternalBInputs) -> float:
    """计算资本存量 K。"""

    interest_rate = safe_positive(external.interest_rate_r, EPS)
    return float(external.capital_income_total) / interest_rate


def compute_l(external: ExternalBInputs) -> float:
    """计算劳动存量 L。"""

    return float(external.labor_population)


def compute_factor_params(external: ExternalBInputs) -> pd.Series:
    """汇总计算 `w,r,K,L`，并返回标准化序列。"""

    out = pd.Series(
        {
            "w": compute_w(external),
            "r": compute_r(external),
            "K": compute_k(external),
            "L": compute_l(external),
        },
        name="value",
        dtype=float,
    )
    out.index.name = "parameter"
    return out

