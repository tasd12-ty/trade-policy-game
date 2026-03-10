"""B 类名义出口额 ExportValue_j 计算。"""

from __future__ import annotations

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .external_inputs import ExternalBInputs


def compute_export_value_j(total_table: IoTableData, external: ExternalBInputs) -> pd.Series:
    """计算部门名义出口额 `ExportValue_j`。"""

    sectors = list(total_table.sector_codes)
    if "EXPO" in total_table.final_demand.columns:
        export_value = (
            total_table.final_demand["EXPO"]
            .reindex(sectors)
            .fillna(external.export_value_fallback)
            .clip(lower=0.0)
        )
    else:
        export_value = pd.Series(external.export_value_fallback, index=sectors, dtype=float)

    export_value.name = "ExportValue_j"
    export_value.index.name = "sector_j"
    return export_value

