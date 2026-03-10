"""C 类消费端 CES 形状参数 rho_cj 计算。"""

from __future__ import annotations

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalCInputs
from .shared import build_sector_series


def compute_rho_cj(total_table: IoTableData, external: ExternalCInputs) -> pd.Series:
    """计算 `rho_cj`（按部门外生给定，可覆盖默认值）。"""

    return build_sector_series(
        total_table.sector_codes,
        default_value=external.rho_cj_default,
        overrides=external.rho_cj_by_sector,
        rho_value=True,
        name="rho_cj",
        index_name="sector_j",
    )

