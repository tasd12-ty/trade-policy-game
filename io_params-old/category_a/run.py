"""CLI entrypoint for A-class parameter extraction from OECD IO tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.constants import DEFAULT_CONSUMPTION_COLUMNS, DEFAULT_DATA_ROOT
from ..common.oecd_io_table import load_country_year_tables
from .alpha_ij import compute_alpha_ij
from .beta_j import compute_beta_j
from .theta_cj import compute_theta_cj
from .theta_ij import compute_theta_ij


def _parse_consumption_columns(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("consumption columns cannot be empty")
    return cols


def _parse_optional_float(raw: str) -> float | None:
    value = raw.strip().lower()
    if value in {"", "none", "nan"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute A-class parameters (alpha_ij, theta_ij, beta_j, theta_cj) from OECD IO CSVs."
    )
    parser.add_argument("--country", required=True, choices=["USA", "CHN"], help="Country code")
    parser.add_argument("--year", required=True, type=int, help="Year, e.g. 2022")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for 中美投入产出表数据（附表格阅读说明）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("io_params/outputs/category_a"),
        help="Output directory",
    )
    parser.add_argument(
        "--consumption-cols",
        type=str,
        default=",".join(DEFAULT_CONSUMPTION_COLUMNS),
        help="Comma-separated final-demand columns used for beta_j/theta_cj",
    )
    parser.add_argument(
        "--theta-ij-fill-on-zero-total",
        type=str,
        default="none",
        help="Fill value when total intermediate use is zero (none|nan|float)",
    )
    parser.add_argument(
        "--theta-cj-fill-on-zero-total",
        type=str,
        default="none",
        help="Fill value when total consumption is zero (none|nan|float)",
    )
    args = parser.parse_args()

    consumption_cols = _parse_consumption_columns(args.consumption_cols)
    theta_ij_fill = _parse_optional_float(args.theta_ij_fill_on_zero_total)
    theta_cj_fill = _parse_optional_float(args.theta_cj_fill_on_zero_total)

    tables = load_country_year_tables(
        data_root=args.data_root,
        country=args.country,
        year=args.year,
    )

    alpha_ij = compute_alpha_ij(tables.total)
    theta_ij = compute_theta_ij(
        total_table=tables.total,
        domestic_table=tables.domestic,
        fill_on_zero_total=theta_ij_fill,
    )
    beta_j = compute_beta_j(
        total_table=tables.total,
        consumption_columns=consumption_cols,
    )
    theta_cj = compute_theta_cj(
        total_table=tables.total,
        domestic_table=tables.domestic,
        consumption_columns=consumption_cols,
        fill_on_zero_total=theta_cj_fill,
    )

    out_base = args.out_dir / f"{args.country}{args.year}"
    out_base.mkdir(parents=True, exist_ok=True)

    alpha_path = out_base / "alpha_ij.csv"
    theta_ij_path = out_base / "theta_ij.csv"
    beta_path = out_base / "beta_j.csv"
    theta_cj_path = out_base / "theta_cj.csv"
    meta_path = out_base / "metadata.json"

    alpha_ij.to_csv(alpha_path, encoding="utf-8")
    theta_ij.to_csv(theta_ij_path, encoding="utf-8")
    beta_j.to_frame().to_csv(beta_path, encoding="utf-8")
    theta_cj.to_frame().to_csv(theta_cj_path, encoding="utf-8")

    metadata = {
        "country": args.country,
        "year": args.year,
        "data_root": str(args.data_root),
        "total_csv": str(tables.total.csv_path),
        "domestic_csv": str(tables.domestic.csv_path),
        "consumption_columns": consumption_cols,
        "theta_ij_fill_on_zero_total": theta_ij_fill,
        "theta_cj_fill_on_zero_total": theta_cj_fill,
        "notes": {
            "alpha_ij": "alpha_ij = Z_total[j,i] / OUTPUT[i], matrix orientation user i x input j",
            "theta_ij": "theta_ij = Z_dom[j,i] / Z_total[j,i], matrix orientation user i x input j",
            "beta_j": "beta_j from normalized selected final-demand spending by product j",
            "theta_cj": "theta_cj = C_dom[j] / C_total[j] over selected final-demand columns",
        },
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"Wrote: {alpha_path}")
    print(f"Wrote: {theta_ij_path}")
    print(f"Wrote: {beta_path}")
    print(f"Wrote: {theta_cj_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()

