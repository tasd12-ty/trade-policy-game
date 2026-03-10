"""C 类参数计算 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..common.constants import DEFAULT_CONSUMPTION_COLUMNS, DEFAULT_DATA_ROOT
from ..common.oecd_io_table import load_country_year_tables
from .external_inputs import load_external_c_inputs
from .pipeline import compute_c_parameters


def _default_external_inputs_path() -> Path:
    return Path(__file__).with_name("external_inputs_default.json")


def _parse_consumption_columns(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("consumption columns cannot be empty")
    return cols


def _load_optional_series(path: Path | None, name: str) -> pd.Series | None:
    """读取可选基线向量 CSV（第一列为索引，第一列数据为值）。"""

    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError(f"{name} CSV is empty: {path}")
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        series = df[name] if name in df.columns else df.iloc[:, 0]
    out = series.astype(float)
    out.name = name
    out.index = out.index.astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute C-class parameters (A_i, gamma_ij, gamma_cj, rho_cj, Export_t, P_t_O)."
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
        default=Path("io_params/outputs/category_c"),
        help="Output directory",
    )
    parser.add_argument(
        "--external-inputs",
        type=Path,
        default=_default_external_inputs_path(),
        help="JSON file containing externally provided C-class values",
    )
    parser.add_argument(
        "--consumption-cols",
        type=str,
        default=",".join(DEFAULT_CONSUMPTION_COLUMNS),
        help="Comma-separated final-demand columns used for gamma_cj base mapping",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Override horizon periods for Export_t/P_t_O path",
    )
    parser.add_argument(
        "--export-base-csv",
        type=Path,
        default=None,
        help="Optional baseline export quantity by sector (index=sector, one value column)",
    )
    parser.add_argument(
        "--p-t-o-base-csv",
        type=Path,
        default=None,
        help="Optional baseline import price by sector (index=sector, one value column)",
    )
    args = parser.parse_args()

    consumption_cols = _parse_consumption_columns(args.consumption_cols)
    tables = load_country_year_tables(
        data_root=args.data_root,
        country=args.country,
        year=args.year,
    )
    external = load_external_c_inputs(args.external_inputs)
    export_base_j = _load_optional_series(args.export_base_csv, "Export_base_j")
    p_t_o_base_j = _load_optional_series(args.p_t_o_base_csv, "P_t_O_base_j")

    outputs = compute_c_parameters(
        total_table=tables.total,
        domestic_table=tables.domestic,
        external=external,
        consumption_columns=consumption_cols,
        export_base_j=export_base_j,
        p_t_o_base_j=p_t_o_base_j,
        periods=args.periods,
    )

    out_base = args.out_dir / f"{args.country}{args.year}"
    out_base.mkdir(parents=True, exist_ok=True)

    a_i_path = out_base / "A_i.csv"
    gamma_ij_path = out_base / "gamma_ij.csv"
    gamma_cj_path = out_base / "gamma_cj.csv"
    rho_cj_path = out_base / "rho_cj.csv"
    export_base_path = out_base / "Export_base_j.csv"
    p_t_o_base_path = out_base / "P_t_O_base_j.csv"
    export_t_path = out_base / "Export_t.csv"
    p_t_o_path = out_base / "P_t_O.csv"
    diagnostics_path = out_base / "diagnostics.json"
    meta_path = out_base / "metadata.json"

    outputs["A_i"].to_frame().to_csv(a_i_path, encoding="utf-8")
    outputs["gamma_ij"].to_csv(gamma_ij_path, encoding="utf-8")
    outputs["gamma_cj"].to_frame().to_csv(gamma_cj_path, encoding="utf-8")
    outputs["rho_cj"].to_frame().to_csv(rho_cj_path, encoding="utf-8")
    outputs["Export_base_j"].to_frame().to_csv(export_base_path, encoding="utf-8")
    outputs["P_t_O_base_j"].to_frame().to_csv(p_t_o_base_path, encoding="utf-8")
    outputs["Export_t"].to_csv(export_t_path, encoding="utf-8")
    outputs["P_t_O"].to_csv(p_t_o_path, encoding="utf-8")

    with diagnostics_path.open("w", encoding="utf-8") as fp:
        json.dump(outputs["diagnostics"], fp, ensure_ascii=False, indent=2)

    metadata = {
        "country": args.country,
        "year": args.year,
        "data_root": str(args.data_root),
        "total_csv": str(tables.total.csv_path),
        "domestic_csv": str(tables.domestic.csv_path),
        "external_inputs_json": str(args.external_inputs),
        "consumption_columns": consumption_cols,
        "periods_override": args.periods,
        "export_base_csv": None if args.export_base_csv is None else str(args.export_base_csv),
        "p_t_o_base_csv": None if args.p_t_o_base_csv is None else str(args.p_t_o_base_csv),
        "notes": {
            "A_i": "外生基线生产率，默认=1，可按部门覆盖",
            "gamma_ij": "按既有口径由 theta_ij 映射，并允许外生覆盖",
            "gamma_cj": "按既有口径由 theta_cj 映射，并允许外生覆盖",
            "rho_cj": "消费端 CES 形状参数，默认=1，可按部门覆盖",
            "Export_t": "Export_t[j] = Export_base_j[j] * multiplier_t[j]",
            "P_t_O": "P_t_O[t,j] = P_t_O_base_j[j] * multiplier_t[j]",
        },
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"Wrote: {a_i_path}")
    print(f"Wrote: {gamma_ij_path}")
    print(f"Wrote: {gamma_cj_path}")
    print(f"Wrote: {rho_cj_path}")
    print(f"Wrote: {export_base_path}")
    print(f"Wrote: {p_t_o_base_path}")
    print(f"Wrote: {export_t_path}")
    print(f"Wrote: {p_t_o_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()

