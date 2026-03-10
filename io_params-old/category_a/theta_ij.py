"""Compute production-side theta_ij from OECD IO tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.oecd_io_table import IoTableData


def compute_theta_ij(
    total_table: IoTableData,
    domestic_table: IoTableData,
    fill_on_zero_total: float | None = None,
) -> pd.DataFrame:
    """Compute theta_ij = Z_dom[j, i] / Z_total[j, i].

    Output orientation follows model notation:
    - index: user sector i
    - columns: input sector j
    """

    if total_table.sector_codes != domestic_table.sector_codes:
        raise ValueError("Sector codes mismatch between total and domestic tables")

    z_total_user_input = total_table.intermediate.T
    z_dom_user_input = domestic_table.intermediate.T.reindex_like(z_total_user_input).fillna(0.0)

    denom = z_total_user_input.replace(0.0, np.nan)
    theta = z_dom_user_input.divide(denom)
    theta = theta.clip(lower=0.0, upper=1.0)
    if fill_on_zero_total is not None:
        theta = theta.fillna(fill_on_zero_total)

    theta.index.name = "user_sector_i"
    theta.columns.name = "input_sector_j"
    return theta

