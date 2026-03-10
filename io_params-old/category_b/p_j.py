"""B 类国内价格 P_j 计算。"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalBInputs
from .shared import EPS, safe_positive


def compute_p_j_exogenous(total_table: IoTableData, external: ExternalBInputs) -> pd.Series:
    """按外生配置计算 `P_j`。"""

    sectors = list(total_table.sector_codes)
    p_j = pd.Series(index=sectors, dtype=float, name="P_j")
    for s in sectors:
        dom = external.domestic_prices_by_sector.get(s, external.domestic_price_default)
        p_j.loc[s] = safe_positive(dom, EPS)
    p_j.index.name = "sector_j"
    return p_j


def compute_p_j_from_endogenous_solution(
    equilibrium_result: Dict[str, Any],
    total_table: IoTableData,
) -> pd.Series:
    """从联立求解结果中提取 `P_j`。"""

    sectors = list(total_table.sector_codes)
    p_j = pd.Series(equilibrium_result["H"]["prices"]["P_j"], index=sectors, dtype=float, name="P_j")
    p_j.index.name = "sector_j"
    return p_j

