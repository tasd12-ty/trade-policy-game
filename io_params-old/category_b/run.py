"""B 类参数计算 CLI（联立求解增强版）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.constants import DEFAULT_CONSUMPTION_COLUMNS, DEFAULT_DATA_ROOT
from ..common.oecd_io_table import load_country_year_tables
from .external_inputs import load_external_b_inputs, load_price_mode
from .pipeline import compute_b_parameters


def _default_external_inputs_path() -> Path:
    return Path(__file__).with_name("external_inputs_default.json")


def _parse_consumption_columns(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("consumption columns cannot be empty")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute B-class parameters (w,r,K,L,P_j,P_j_O,E_j) from IO + external inputs."
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
        default=Path("io_params/outputs/category_b"),
        help="Output directory",
    )
    parser.add_argument(
        "--external-inputs",
        type=Path,
        default=_default_external_inputs_path(),
        help="JSON file containing externally provided B-class values",
    )
    parser.add_argument(
        "--price-mode",
        type=str,
        choices=["endogenous", "exogenous", "auto"],
        default="auto",
        help="Price calibration mode; auto means read from external-inputs JSON",
    )
    parser.add_argument(
        "--consumption-cols",
        type=str,
        default=",".join(DEFAULT_CONSUMPTION_COLUMNS),
        help="Comma-separated final-demand columns used in endogenous price calibration",
    )
    parser.add_argument(
        "--solver-max-iterations",
        type=int,
        default=400,
        help="Max iterations for endogenous equilibrium solver",
    )
    parser.add_argument(
        "--solver-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for endogenous equilibrium solver",
    )
    args = parser.parse_args()

    consumption_cols = _parse_consumption_columns(args.consumption_cols)
    tables = load_country_year_tables(
        data_root=args.data_root,
        country=args.country,
        year=args.year,
    )
    external = load_external_b_inputs(args.external_inputs)
    if args.price_mode == "auto":
        price_mode = load_price_mode(args.external_inputs)
    else:
        price_mode = args.price_mode
    outputs = compute_b_parameters(
        total_table=tables.total,
        domestic_table=tables.domestic,
        external=external,
        consumption_columns=consumption_cols,
        price_mode=price_mode,
        solver_max_iterations=args.solver_max_iterations,
        solver_tolerance=args.solver_tolerance,
    )

    out_base = args.out_dir / f"{args.country}{args.year}"
    out_base.mkdir(parents=True, exist_ok=True)

    factor_path = out_base / "factor_params.csv"
    p_j_path = out_base / "P_j.csv"
    p_j_o_path = out_base / "P_j_O.csv"
    e_j_path = out_base / "E_j.csv"
    export_value_path = out_base / "ExportValue_j.csv"
    diagnostics_path = out_base / "diagnostics.json"
    meta_path = out_base / "metadata.json"

    outputs["factor_params"].to_frame().to_csv(factor_path, encoding="utf-8")
    outputs["P_j"].to_frame().to_csv(p_j_path, encoding="utf-8")
    outputs["P_j_O"].to_frame().to_csv(p_j_o_path, encoding="utf-8")
    outputs["E_j"].to_frame().to_csv(e_j_path, encoding="utf-8")
    outputs["ExportValue_j"].to_frame().to_csv(export_value_path, encoding="utf-8")
    with diagnostics_path.open("w", encoding="utf-8") as fp:
        json.dump(outputs["diagnostics"], fp, ensure_ascii=False, indent=2)

    metadata = {
        "country": args.country,
        "year": args.year,
        "data_root": str(args.data_root),
        "total_csv": str(tables.total.csv_path),
        "domestic_csv": str(tables.domestic.csv_path),
        "external_inputs_json": str(args.external_inputs),
        "price_mode": price_mode,
        "consumption_columns": consumption_cols,
        "solver_max_iterations": int(args.solver_max_iterations),
        "solver_tolerance": float(args.solver_tolerance),
        "notes": {
            "factor_params": "w=LaborIncome/L, r=external, K=CapitalIncome/r, L=external labor population",
            "P_j": "Endogenous mode: solved from static equilibrium system; exogenous mode: from external JSON",
            "P_j_O": "Endogenous mode: implied import prices from equilibrium; exogenous mode: from external JSON",
            "E_j": "Baseline export quantity E_j = ExportValue_j / P_j",
            "status": "Enhanced B-class implementation with endogenous price calibration and exogenous fallback",
        },
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"Wrote: {factor_path}")
    print(f"Wrote: {p_j_path}")
    print(f"Wrote: {p_j_o_path}")
    print(f"Wrote: {e_j_path}")
    print(f"Wrote: {export_value_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
