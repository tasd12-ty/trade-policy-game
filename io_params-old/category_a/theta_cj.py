"""Compute consumption-side theta_cj from OECD IO tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.oecd_io_table import IoTableData


def compute_theta_cj(
    total_table: IoTableData,
    domestic_table: IoTableData,
    consumption_columns: list[str],
    fill_on_zero_total: float | None = None,
) -> pd.Series:
    """Compute theta_cj = C_dom[j] / C_total[j] for selected final-demand columns."""

    if total_table.sector_codes != domestic_table.sector_codes:
        raise ValueError("Sector codes mismatch between total and domestic tables")

    missing_total = [c for c in consumption_columns if c not in total_table.final_demand.columns]
    missing_dom = [c for c in consumption_columns if c not in domestic_table.final_demand.columns]
    if missing_total:
        raise ValueError(f"Missing final-demand columns in total table: {missing_total}")
    if missing_dom:
        raise ValueError(f"Missing final-demand columns in domestic table: {missing_dom}")

    c_total = total_table.final_demand[consumption_columns].sum(axis=1)
    c_dom = domestic_table.final_demand[consumption_columns].sum(axis=1).reindex(c_total.index).fillna(0.0)

    denom = c_total.replace(0.0, np.nan)
    theta_cj = (c_dom / denom).clip(lower=0.0, upper=1.0)
    if fill_on_zero_total is not None:
        theta_cj = theta_cj.fillna(fill_on_zero_total)

    theta_cj.name = "theta_cj"
    theta_cj.index.name = "product_sector_j"
    return theta_cj

