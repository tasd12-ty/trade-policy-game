"""Convert 5-sector Chinese non-competitive IO tables to OECD CSV format.

Reads CHN2017_5sector_CNlike.xlsx and USA2017_5sector_CNlike.xlsx,
produces TTL/DOM CSV files compatible with the io_params pipeline.

Sector mapping:
    A01  -> D01 (Agriculture/soybean)
    RUP  -> D02 (Rare earth upstream)
    RMID -> D03 (Rare earth midstream)
    ELEC -> D04 (Electronics/ICT)
    TEQ  -> D05 (Auto/machinery)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Configuration ----

SECTOR_CODES = ["D01", "D02", "D03", "D04", "D05"]
SECTOR_NAMES = ["A01", "RUP", "RMID", "ELEC", "TEQ"]

SRC_DIR = Path("usa2017_5sector_cnlike")
DST_ROOT = Path("中美投入产出表数据（附表格阅读说明）")

FILES = {
    "CHN": "CHN2017_5sector_CNlike.xlsx",
    "USA": "USA2017_5sector_CNlike.xlsx",
}
YEAR = 2017

# Row indices in the Excel main sheet (0-based, after title/unit rows)
# Row 0-1: title + units
# Row 2-4: headers (merged)
# Row 5-9: domestic intermediate (5 sectors)
# Row 10-14: import intermediate (5 sectors)
# Row 15: TII (total intermediate inputs)
# Row 16-19: VA001-VA004
# Row 20: TVA (total value added)
# Row 21: TI (total input = gross output)

DOM_ROWS = slice(5, 10)    # domestic intermediate
IMP_ROWS = slice(10, 15)   # import intermediate
VA_ROWS = slice(16, 20)    # value added components
TVA_ROW = 20               # total value added
TI_ROW = 21                # total input (= gross output)

# Column indices (0-based) in the raw Excel
INT_COLS = slice(4, 9)     # 5x5 intermediate use matrix
TC_COL = 10                # total consumption
GFCF_COL = 11              # gross fixed capital formation
INVNT_COL = 12             # inventory change
EXPO_COL = 13              # exports
GO_COL = 15                # gross output (domestic) / imports (import rows)


def read_io_table(xlsx_path: str | Path) -> dict:
    """Read the 5-sector IO table from Excel.

    Returns dict with:
        dom_Z: (5, 5) domestic intermediate matrix
        imp_Z: (5, 5) import intermediate matrix
        dom_fd: (5, 4) domestic final demand [HFCE, GFCF, INVNT, EXPO]
        imp_fd: (5, 4) import final demand
        va: (5,) total value added per sector
        output: (5,) gross output per sector
        imp_total: (5,) total imports per sector
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    # Extract numeric data
    data = df.values

    # Domestic intermediate matrix (supplier j x user i)
    dom_Z = np.array(data[DOM_ROWS, INT_COLS], dtype=float)
    # Import intermediate matrix
    imp_Z = np.array(data[IMP_ROWS, INT_COLS], dtype=float)

    # Final demand columns for domestic rows
    dom_fd = np.zeros((5, 4), dtype=float)
    dom_fd[:, 0] = np.array(data[DOM_ROWS, TC_COL], dtype=float)   # HFCE
    dom_fd[:, 1] = np.array(data[DOM_ROWS, GFCF_COL], dtype=float)  # GFCF
    dom_fd[:, 2] = np.array(data[DOM_ROWS, INVNT_COL], dtype=float)  # INVNT
    dom_fd[:, 3] = np.array(data[DOM_ROWS, EXPO_COL], dtype=float)   # EXPO

    # Final demand columns for import rows
    imp_fd = np.zeros((5, 4), dtype=float)
    imp_fd[:, 0] = np.array(data[IMP_ROWS, TC_COL], dtype=float)
    imp_fd[:, 1] = np.array(data[IMP_ROWS, GFCF_COL], dtype=float)
    imp_fd[:, 2] = np.array(data[IMP_ROWS, INVNT_COL], dtype=float)
    imp_fd[:, 3] = np.array(data[IMP_ROWS, EXPO_COL], dtype=float)

    # Total value added per sector (from TVA row)
    va = np.array(data[TVA_ROW, INT_COLS], dtype=float)

    # Gross output per sector (from TI row, intermediate columns)
    output = np.array(data[TI_ROW, INT_COLS], dtype=float)

    # Total imports per sector (from GO/IM column of import rows)
    imp_total = np.array(data[IMP_ROWS, GO_COL], dtype=float)

    return {
        "dom_Z": dom_Z,
        "imp_Z": imp_Z,
        "dom_fd": dom_fd,
        "imp_fd": imp_fd,
        "va": va,
        "output": output,
        "imp_total": imp_total,
    }


def write_oecd_csv(
    path: str | Path,
    Z: np.ndarray,
    fd: np.ndarray,
    va: np.ndarray,
    output: np.ndarray,
    row_prefix: str,
) -> None:
    """Write an OECD-format IO CSV.

    Args:
        path: output CSV path
        Z: (5, 5) intermediate matrix (supplier x user)
        fd: (5, 4) final demand [HFCE, GFCF, INVNT, EXPO]
        va: (5,) value added
        output: (5,) gross output
        row_prefix: "TTL_" or "DOM_"
    """
    fd_cols = ["HFCE", "GFCF", "INVNT", "EXPO"]
    all_cols = SECTOR_CODES + fd_cols

    rows = []

    # Sector rows
    for i, code in enumerate(SECTOR_CODES):
        # Row label: TTL_01 or DOM_01
        suffix = code[1:]  # "01", "02", etc.
        row_label = f"{row_prefix}{suffix}"
        values = list(Z[i, :]) + list(fd[i, :])
        rows.append([row_label] + values)

    # VALU row (value added)
    valu_row = ["VALU"] + list(va) + [0.0] * len(fd_cols)
    rows.append(valu_row)

    # OUTPUT row
    output_row = ["OUTPUT"] + list(output) + [0.0] * len(fd_cols)
    rows.append(output_row)

    df = pd.DataFrame(rows, columns=[""] + all_cols)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Written: {path}")


def convert_country(country: str, xlsx_path: Path, dst_root: Path) -> None:
    """Convert one country's Excel to OECD TTL + DOM CSVs."""
    print(f"\n{'='*60}")
    print(f"Converting {country} from {xlsx_path}")
    print(f"{'='*60}")

    tbl = read_io_table(xlsx_path)

    # Print summary
    print(f"\n  Domestic Z (5x5):\n{tbl['dom_Z']}")
    print(f"\n  Import Z (5x5):\n{tbl['imp_Z']}")
    print(f"\n  Gross output: {tbl['output']}")
    print(f"  Value added:  {tbl['va']}")
    print(f"  Total imports: {tbl['imp_total']}")

    # Total = domestic + import
    ttl_Z = tbl["dom_Z"] + tbl["imp_Z"]
    ttl_fd = tbl["dom_fd"] + tbl["imp_fd"]

    # Sanity checks
    for i, code in enumerate(SECTOR_CODES):
        int_use = ttl_Z[:, i].sum()
        fd_use = ttl_fd[i, :].sum()  # final demand of product i
        supply_side = tbl["va"][i] + ttl_Z[:, i].sum()  # VA + intermediate inputs
        print(f"\n  {code}: output={tbl['output'][i]:.1f}, "
              f"int_inputs={int_use:.1f}, VA={tbl['va'][i]:.1f}, "
              f"VA+int={supply_side:.1f}")

    # Create output directories
    country_dir_map = {
        "CHN": "中国/OECD中国投入产出数据",
        "USA": "美国/OECD美国投入产出数据",
    }
    base = dst_root / country_dir_map[country]
    ttl_dir = base / "投入产出，行业-行业，total口径"
    dom_dir = base / "投入产出（含进口行），行业-行业，Domestic口径"

    ttl_dir.mkdir(parents=True, exist_ok=True)
    dom_dir.mkdir(parents=True, exist_ok=True)

    # Write TTL CSV
    ttl_path = ttl_dir / f"{country}{YEAR}ttl.csv"
    write_oecd_csv(ttl_path, ttl_Z, ttl_fd, tbl["va"], tbl["output"], "TTL_")

    # Write DOM CSV
    dom_path = dom_dir / f"{country}{YEAR}dom.csv"
    write_oecd_csv(dom_path, tbl["dom_Z"], tbl["dom_fd"], tbl["va"], tbl["output"], "DOM_")

    # Verification: read back and check
    print(f"\n  --- Verification ---")
    for label, fpath in [("TTL", ttl_path), ("DOM", dom_path)]:
        df_check = pd.read_csv(fpath, encoding="utf-8-sig", index_col=0)
        print(f"  {label}: shape={df_check.shape}, columns={list(df_check.columns)}")
        print(f"  {label} index: {list(df_check.index)}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for country, fname in FILES.items():
        xlsx_path = SRC_DIR / fname
        if not xlsx_path.exists():
            print(f"SKIP: {xlsx_path} not found")
            continue
        convert_country(country, xlsx_path, DST_ROOT)

    print("\n\nDone! Files ready for io_params pipeline.")
    print("Run with: --data-root '中美投入产出表数据（附表格阅读说明）' --country CHN --year 2017")
    print("      or: --data-root '中美投入产出表数据（附表格阅读说明）' --country USA --year 2017")


if __name__ == "__main__":
    main()
