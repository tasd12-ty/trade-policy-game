"""C 类生产端 Armington 权重 gamma_ij 计算。"""

from __future__ import annotations

import pandas as pd

from ..category_a.theta_ij import compute_theta_ij
from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalCInputs
from .shared import safe_probability


def compute_gamma_ij(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalCInputs,
) -> pd.DataFrame:
    """按既有口径由 `theta_ij` 映射得到 `gamma_ij`，再叠加外生覆盖。"""

    gamma = compute_theta_ij(
        total_table=total_table,
        domestic_table=domestic_table,
        fill_on_zero_total=safe_probability(external.gamma_ij_fill_on_zero_total, 1.0),
    ).copy()

    # 外生覆盖项：外层键是 user i，内层键是 input j。
    for i, row in external.gamma_ij_by_user_input.items():
        if i not in gamma.index:
            continue
        for j, v in row.items():
            if j not in gamma.columns:
                continue
            gamma.loc[i, j] = safe_probability(v, external.gamma_ij_fill_on_zero_total)

    low = min(external.gamma_ij_clip_min, external.gamma_ij_clip_max)
    high = max(external.gamma_ij_clip_min, external.gamma_ij_clip_max)
    gamma = gamma.clip(lower=low, upper=high)
    gamma.index.name = "user_sector_i"
    gamma.columns.name = "input_sector_j"
    return gamma

