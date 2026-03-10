"""B 类进口价格 P_j^O 计算。"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalBInputs
from .shared import EPS, safe_positive


def compute_p_j_o_exogenous(
    total_table: IoTableData,
    external: ExternalBInputs,
    p_j: pd.Series,
) -> pd.Series:
    """按外生配置计算 `P_j^O`。"""

    sectors = list(total_table.sector_codes)
    p_j_o = pd.Series(index=sectors, dtype=float, name="P_j_O")
    for s in sectors:
        if s in external.import_prices_by_sector:
            imp = external.import_prices_by_sector[s]
        else:
            imp = float(p_j.loc[s]) * float(external.import_price_multiplier_default)
        p_j_o.loc[s] = safe_positive(imp, EPS)
    p_j_o.index.name = "sector_j"
    return p_j_o


def compute_p_j_o_from_endogenous_solution(
    equilibrium_result: Dict[str, Any],
    total_table: IoTableData,
) -> pd.Series:
    """从联立求解结果中提取 `P_j^O`。"""

    sectors = list(total_table.sector_codes)
    p_j_o = pd.Series(equilibrium_result["H"]["prices"]["P_O_j"], index=sectors, dtype=float, name="P_j_O")
    p_j_o.index.name = "sector_j"
    return p_j_o

