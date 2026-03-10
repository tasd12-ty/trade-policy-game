"""Compute alpha_ij from OECD IO tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.oecd_io_table import IoTableData


def compute_alpha_ij(total_table: IoTableData) -> pd.DataFrame:
    """Compute alpha_ij using formula alpha_ij = Z[j, i] / OUTPUT[i].

    Output orientation follows model notation:
    - index: user sector i
    - columns: input sector j
    """

    z_supplier_user = total_table.intermediate
    output_user = total_table.output

    z_user_input = z_supplier_user.T
    denom = output_user.reindex(z_user_input.index).replace(0.0, np.nan)
    alpha = z_user_input.divide(denom, axis=0)
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alpha.index.name = "user_sector_i"
    alpha.columns.name = "input_sector_j"
    return alpha

