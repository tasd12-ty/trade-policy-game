"""IO table loader utilities for OECD total/domestic CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .constants import (
    COUNTRY_DIR,
    DOMESTIC_PREFIX,
    DOMESTIC_SUBDIR,
    OUTPUT_ROW_LABEL,
    TOTAL_PREFIX,
    TOTAL_SUBDIR,
)


@dataclass(frozen=True)
class IoTableData:
    """Parsed data from a single OECD IO CSV."""

    country: str
    year: int
    table_type: str
    sector_codes: List[str]
    intermediate: pd.DataFrame
    final_demand: pd.DataFrame
    output: pd.Series
    csv_path: Path


@dataclass(frozen=True)
class CountryYearTables:
    """Pair of total + domestic tables for a country-year."""

    total: IoTableData
    domestic: IoTableData


def _is_sector_code(value: str) -> bool:
    return value.startswith("D") and any(ch.isdigit() for ch in value)


def _canonicalize_row_sector(row_label: str, prefix: str) -> str | None:
    if not row_label.startswith(prefix):
        return None
    suffix = row_label[len(prefix) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return None
    return suffix if suffix.startswith("D") else f"D{suffix}"


def _coerce_numeric(df: pd.DataFrame, skip_columns: Iterable[str]) -> pd.DataFrame:
    skip = set(skip_columns)
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _extract_table(
    country: str,
    year: int,
    csv_path: Path,
    table_type: str,
    row_prefix: str,
) -> IoTableData:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError(f"CSV has no rows: {csv_path}")

    row_label_col = raw.columns[0]
    raw[row_label_col] = raw[row_label_col].astype(str)
    df = _coerce_numeric(raw, skip_columns=[row_label_col])

    sector_codes = [c for c in df.columns if _is_sector_code(c)]
    if not sector_codes:
        raise ValueError(f"No sector columns found in {csv_path}")

    row_to_values: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        row_label = str(row[row_label_col])
        code = _canonicalize_row_sector(row_label, row_prefix)
        if code is None:
            continue
        if code in sector_codes:
            row_to_values[code] = row

    missing_rows = [s for s in sector_codes if s not in row_to_values]
    if missing_rows:
        raise ValueError(
            f"Missing {len(missing_rows)} sector rows in {csv_path}: "
            f"{missing_rows[:5]}{'...' if len(missing_rows) > 5 else ''}"
        )

    # Supplier-by-user matrix Z[j, i], where row is supplier product j and column is user i.
    intermediate = pd.DataFrame(
        {sector: row_to_values[sector][sector_codes].astype(float) for sector in sector_codes}
    ).T
    intermediate.index.name = "supplier_sector"
    intermediate.columns.name = "user_sector"

    output_rows = df[df[row_label_col] == OUTPUT_ROW_LABEL]
    if output_rows.empty:
        raise ValueError(f"Row '{OUTPUT_ROW_LABEL}' not found in {csv_path}")
    output = output_rows.iloc[0][sector_codes].astype(float)
    output.index.name = "user_sector"
    output.name = "output"

    final_demand_cols = [c for c in df.columns if c not in {row_label_col, *sector_codes}]
    final_demand = pd.DataFrame(
        {sector: row_to_values[sector][final_demand_cols].astype(float) for sector in sector_codes}
    ).T
    final_demand.index.name = "product_sector"

    return IoTableData(
        country=country,
        year=year,
        table_type=table_type,
        sector_codes=sector_codes,
        intermediate=intermediate,
        final_demand=final_demand,
        output=output,
        csv_path=csv_path,
    )


def _build_csv_paths(data_root: Path, country: str, year: int) -> tuple[Path, Path]:
    country = country.upper()
    if country not in COUNTRY_DIR:
        raise ValueError(f"Unsupported country '{country}', expected one of {sorted(COUNTRY_DIR)}")

    base = data_root / COUNTRY_DIR[country]
    total = base / TOTAL_SUBDIR / f"{country}{year}ttl.csv"
    domestic = base / DOMESTIC_SUBDIR / f"{country}{year}dom.csv"
    return total, domestic


def load_country_year_tables(data_root: Path, country: str, year: int) -> CountryYearTables:
    """Load total + domestic OECD IO tables for one country and year."""

    country = country.upper()
    total_path, domestic_path = _build_csv_paths(data_root, country, year)

    total = _extract_table(
        country=country,
        year=year,
        csv_path=total_path,
        table_type="total",
        row_prefix=TOTAL_PREFIX,
    )
    domestic = _extract_table(
        country=country,
        year=year,
        csv_path=domestic_path,
        table_type="domestic",
        row_prefix=DOMESTIC_PREFIX,
    )

    if total.sector_codes != domestic.sector_codes:
        raise ValueError("Sector codes mismatch between total and domestic tables")

    return CountryYearTables(total=total, domestic=domestic)

