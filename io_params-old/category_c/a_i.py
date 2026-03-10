"""C 类基线生产率 A_i 计算。"""

from __future__ import annotations

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalCInputs
from .shared import build_sector_series


def compute_a_i(total_table: IoTableData, external: ExternalCInputs) -> pd.Series:
    """计算 `A_i`（按部门基线外生给定）。"""

    return build_sector_series(
        total_table.sector_codes,
        default_value=external.a_i_default,
        overrides=external.a_i_by_sector,
        positive=True,
        name="A_i",
        index_name="sector_i",
    )

