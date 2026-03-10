"""B 类基期出口量 E_j 计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_e_j(export_value_j: pd.Series, p_j: pd.Series) -> pd.Series:
    """计算 `E_j = ExportValue_j / P_j`。"""

    denom = p_j.replace(0.0, np.nan)
    e_j = (export_value_j / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    e_j.name = "E_j"
    e_j.index.name = "sector_j"
    return e_j

