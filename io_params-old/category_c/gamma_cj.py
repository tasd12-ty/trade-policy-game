"""C 类消费端 Armington 权重 gamma_cj 计算。"""

from __future__ import annotations

import pandas as pd

from ..category_a.theta_cj import compute_theta_cj
from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalCInputs
from .shared import safe_probability


def compute_gamma_cj(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalCInputs,
    consumption_columns: list[str],
) -> pd.Series:
    """按既有口径由 `theta_cj` 映射得到 `gamma_cj`，再叠加外生覆盖。"""

    gamma_cj = compute_theta_cj(
        total_table=total_table,
        domestic_table=domestic_table,
        consumption_columns=consumption_columns,
        fill_on_zero_total=safe_probability(external.gamma_cj_fill_on_zero_total, 1.0),
    ).copy()
    gamma_cj.name = "gamma_cj"

    for s, v in external.gamma_cj_by_sector.items():
        if s not in gamma_cj.index:
            continue
        gamma_cj.loc[s] = safe_probability(v, external.gamma_cj_fill_on_zero_total)

    low = min(external.gamma_cj_clip_min, external.gamma_cj_clip_max)
    high = max(external.gamma_cj_clip_min, external.gamma_cj_clip_max)
    gamma_cj = gamma_cj.clip(lower=low, upper=high)
    gamma_cj.index.name = "sector_j"
    return gamma_cj

