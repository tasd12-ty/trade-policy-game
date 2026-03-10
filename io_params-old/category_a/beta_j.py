"""Compute beta_j from final demand in OECD IO tables."""

from __future__ import annotations

import pandas as pd

from ..common.oecd_io_table import IoTableData


def compute_beta_j(
    total_table: IoTableData,
    consumption_columns: list[str],
) -> pd.Series:
    """Compute beta_j as normalized final-consumption share by product j."""

    missing = [c for c in consumption_columns if c not in total_table.final_demand.columns]
    if missing:
        raise ValueError(f"Missing final-demand columns in total table: {missing}")

    spending = total_table.final_demand[consumption_columns].sum(axis=1).clip(lower=0.0)
    total_spending = float(spending.sum())
    if total_spending <= 0.0:
        raise ValueError("Total selected final consumption is zero; cannot normalize beta_j")

    beta = spending / total_spending
    beta.name = "beta_j"
    beta.index.name = "product_sector_j"
    return beta

