#!/usr/bin/env python3
"""
Minimal OECD (USA) IO -> eco_simu params ETL (KISS).

What it does (domestic + total):
- Parse OECD Industry-by-Industry CSVs (USA20xxdom.csv and USA20xxttl.csv)
- Build intermediate use matrices Z_dom (DOM_* rows, D* columns) and Z_tot (TTL_* rows, D* columns)
- Read OUTPUT (row) by industry, HFCE by product, EXPO by product
- Aggregate to 6 sectors (optional, default) with a simple mapping
- Calibrate model parameters for one country block (CountryParams-compatible dict):
  * alpha[i,j] = Z_tot[j,i] / OUTPUT[i]
  * beta[j] = HFCE[j] / sum(HFCE)
  * gamma_ij = gamma_j (constant across i), where gamma_prod_j = sum_i Z_dom[j,i] / sum_i Z_tot[j,i]
    - For consumption: gamma_cons[j] = HFCE_dom[j] / HFCE_tot[j] (fallback to gamma_prod if missing)
  * rho (prod/cons) = constant (default 0.2)
  * import_cost[j] = 1.0
  * A[i] = 1.0 (TFP will be identified by model solver)
  * exports[j] = max(EXPO[j], 0.0)

Notes:
- We use TOTAL table for alpha so imported and domestic inputs are both counted in production cost shares.
- Value-added share becomes 1 - sum_j alpha[i,j] inside the model; ensure row-sum < 1 (we clamp if needed).
- Mapping is heuristic; adjust to your project’s aggregation via --mapping-json if needed.

Usage examples:
  python scripts/etl_oecd_to_params.py \
      --oecd-dir "中美投入产出表数据（附表格阅读说明）/美国/OECD美国投入产出数据" \
      --year 2019 \
      --aggregate-6 \
      --out params_USA2019_agg6.json

The output JSON contains a single-country block under key "USA" and a convenience two-country
dict with both H and F set to the same block (for quick smoke tests only).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = rows[1:]
    return header, data


def _is_sector_code(c: str) -> bool:
    return c.startswith('D') and any(ch.isdigit() for ch in c)


def load_oecd_us_tables(oecd_dir: Path, year: int) -> Dict[str, Dict]:
    # Files
    dom = oecd_dir / f"投入产出（含进口行），行业-行业，Domestic口径/USA{year}dom.csv"
    ttl = oecd_dir / f"投入产出，行业-行业，total口径/USA{year}ttl.csv"
    if not dom.exists() or not ttl.exists():
        raise FileNotFoundError(f"Missing USA{year} dom/ttl CSVs under {oecd_dir}")

    # Total table (preferred for alpha & OUTPUT/HFCE/EXPO)
    h_ttl, rows_ttl = _read_csv(ttl)
    # Column indices
    col_idx_ttl = {k: i for i, k in enumerate(h_ttl)}
    # Collect sector columns (buyers)
    sector_cols = [k for k in h_ttl if _is_sector_code(k)]

    # Build Z_tot: rows TTL_* (products), cols D* (industries)
    Z_tot: Dict[str, Dict[str, float]] = {}
    OUTPUT: Dict[str, float] = {}
    HFCE_tot: Dict[str, float] = {}
    EXPO: Dict[str, float] = {}

    for row in rows_ttl:
        rid = row[0]
        if rid.startswith('TTL_'):
            code = rid.replace('TTL_', 'D')
            # totals across industry columns
            Z_tot[code] = {}
            for c in sector_cols:
                try:
                    Z_tot[code][c] = float(row[col_idx_ttl[c]])
                except ValueError:
                    Z_tot[code][c] = 0.0
            # Final demand slices we need
            def _get(name: str) -> float:
                try:
                    return float(row[col_idx_ttl[name]])
                except Exception:
                    return 0.0
            HFCE_tot[code] = _get('HFCE')
            EXPO[code] = max(_get('EXPO'), 0.0)
        elif rid == 'OUTPUT':
            for c in sector_cols:
                try:
                    OUTPUT[c] = float(row[col_idx_ttl[c]])
                except ValueError:
                    OUTPUT[c] = 0.0

    # Domestic table to split domestic/import shares
    h_dom, rows_dom = _read_csv(dom)
    col_idx_dom = {k: i for i, k in enumerate(h_dom)}
    sector_cols_dom = [k for k in h_dom if _is_sector_code(k)]
    # Sanity
    if set(sector_cols_dom) != set(sector_cols):
        # Fallback: keep intersection
        sector_cols = [c for c in sector_cols if c in sector_cols_dom]

    Z_dom: Dict[str, Dict[str, float]] = {}
    HFCE_dom: Dict[str, float] = {}
    for row in rows_dom:
        rid = row[0]
        if rid.startswith('DOM_'):
            code = rid.replace('DOM_', 'D')
            Z_dom[code] = {}
            for c in sector_cols:
                try:
                    Z_dom[code][c] = float(row[col_idx_dom[c]])
                except ValueError:
                    Z_dom[code][c] = 0.0
            # Final demand slice
            try:
                HFCE_dom[code] = float(row[col_idx_dom['HFCE']])
            except Exception:
                HFCE_dom[code] = 0.0

    return {
        'sector_cols': sector_cols,
        'Z_tot': Z_tot,
        'Z_dom': Z_dom,
        'OUTPUT': OUTPUT,
        'HFCE_tot': HFCE_tot,
        'HFCE_dom': HFCE_dom,
        'EXPO': EXPO,
    }


def default_oecd50_to_agg6() -> Dict[str, int]:
    """Heuristic mapping D-codes -> 6 groups.
    Groups (index -> label):
      0 Agriculture (D01, D02, D03)
      1 Mining & quarrying (D05..D09)
      2 Manufacturing (D10T12, D13T15, D16, D17T18, D19, D20, D21, D22, D23, D24A, D24B, D25, D26, D27, D28, D29, D301, D302T309, D31T33)
      3 Utilities & Construction (D35, D36T39, D41T43)
      4 Trade, Transport, Hotels (D45T47, D49, D50, D51, D52, D53, D55T56)
      5 Other services (D58T60, D61, D62T63, D64T66, D68, D69T75, D77T82, D84, D85, D86T88, D90T93, D94T96, D97T98)
    """
    g: Dict[str, int] = {}
    # 0 Agriculture
    for k in ['D01', 'D02', 'D03']:
        g[k] = 0
    # 1 Mining
    for k in ['D05', 'D06', 'D07', 'D08', 'D09']:
        g[k] = 1
    # 2 Manufacturing
    manuf = [
        'D10T12','D13T15','D16','D17T18','D19','D20','D21','D22','D23','D24A','D24B','D25','D26','D27','D28','D29','D301','D302T309','D31T33'
    ]
    for k in manuf:
        g[k] = 2
    # 3 Utilities & Construction
    for k in ['D35','D36T39','D41T43']:
        g[k] = 3
    # 4 Trade/Transport/Hotels
    for k in ['D45T47','D49','D50','D51','D52','D53','D55T56']:
        g[k] = 4
    # 5 Other services
    others = ['D58T60','D61','D62T63','D64T66','D68','D69T75','D77T82','D84','D85','D86T88','D90T93','D94T96','D97T98']
    for k in others:
        g[k] = 5
    return g


def aggregate_to_6(sector_cols: List[str], Z: Dict[str, Dict[str, float]], vecs: Dict[str, Dict[str, float]], mapping: Dict[str, int]):
    n = 6
    # Build index lists per group
    groups: List[List[str]] = [[] for _ in range(n)]
    for s in sector_cols:
        if s in mapping:
            groups[mapping[s]].append(s)
    # Z_agg[jg][ig]
    Z_agg = [[0.0 for _ in range(n)] for __ in range(n)]
    for js, row in Z.items():
        if js not in mapping:
            continue
        jg = mapping[js]
        for isec, val in row.items():
            if isec not in mapping:
                continue
            ig = mapping[isec]
            Z_agg[jg][ig] += val
    # vecs: dict name -> product vector {code->val}
    vecs_agg: Dict[str, List[float]] = {}
    for name, vec in vecs.items():
        v = [0.0 for _ in range(n)]
        for code, val in vec.items():
            if code in mapping:
                v[mapping[code]] += val
        vecs_agg[name] = v
    return Z_agg, vecs_agg


def build_country_params_from_oecd(oecd: Dict, aggregate6: bool = True) -> Dict:
    sector_cols: List[str] = oecd['sector_cols']
    Z_tot: Dict[str, Dict[str, float]] = oecd['Z_tot']
    Z_dom: Dict[str, Dict[str, float]] = oecd['Z_dom']
    OUTPUT: Dict[str, float] = oecd['OUTPUT']
    HFCE_tot: Dict[str, float] = oecd['HFCE_tot']
    HFCE_dom: Dict[str, float] = oecd['HFCE_dom']
    EXPO: Dict[str, float] = oecd['EXPO']

    if aggregate6:
        mapping = default_oecd50_to_agg6()
        Ztot_agg, vecs_agg = aggregate_to_6(
            sector_cols,
            Z_tot,
            {
                'OUTPUT': OUTPUT,
                'HFCE_tot': HFCE_tot,
                'HFCE_dom': HFCE_dom,
                'EXPO': EXPO,
            },
            mapping,
        )
        Zdom_agg, _ = aggregate_to_6(sector_cols, Z_dom, {}, mapping)
        OUTPUT_v = vecs_agg['OUTPUT']
        HFCE_tot_v = vecs_agg['HFCE_tot']
        HFCE_dom_v = vecs_agg['HFCE_dom']
        EXPO_v = vecs_agg['EXPO']
        n = 6
    else:
        # No aggregation: preserve native sector order from sector_cols
        idx = {s: i for i, s in enumerate(sector_cols)}
        n = len(sector_cols)
        Ztot_agg = [[0.0 for _ in range(n)] for __ in range(n)]
        Zdom_agg = [[0.0 for _ in range(n)] for __ in range(n)]
        for js, row in Z_tot.items():
            if js not in idx:
                continue
            j = idx[js]
            for isec, val in row.items():
                if isec in idx:
                    i = idx[isec]
                    Ztot_agg[j][i] = val
        for js, row in Z_dom.items():
            if js not in idx:
                continue
            j = idx[js]
            for isec, val in row.items():
                if isec in idx:
                    i = idx[isec]
                    Zdom_agg[j][i] = val
        OUTPUT_v = [OUTPUT.get(s, 0.0) for s in sector_cols]
        HFCE_tot_v = [HFCE_tot.get(s, 0.0) for s in sector_cols]
        HFCE_dom_v = [HFCE_dom.get(s, 0.0) for s in sector_cols]
        EXPO_v = [EXPO.get(s, 0.0) for s in sector_cols]

    # Build alpha[i,j]
    alpha = [[0.0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        denom = OUTPUT_v[i] if OUTPUT_v[i] > 0 else 1.0
        for j in range(n):
            alpha[i][j] = Ztot_agg[j][i] / denom
    # Clamp row sums to below 1.0, preserving proportions (avoid zero VA share)
    for i in range(n):
        s = sum(alpha[i])
        if s >= 0.98:
            scale = 0.98 / s
            alpha[i] = [a * scale for a in alpha[i]]

    # Beta (HFCE)
    total_hfce = sum(HFCE_tot_v)
    if total_hfce <= 0:
        beta = [1.0 / n] * n
    else:
        beta = [max(x, 0.0) / total_hfce for x in HFCE_tot_v]

    # Gamma (prod & cons)
    gamma_prod = []
    gamma_cons = []
    for j in range(n):
        tot_use = sum(Ztot_agg[j])
        dom_use = sum(Zdom_agg[j])
        if tot_use <= 1e-9:
            gamma_prod.append(1.0)
        else:
            gamma_prod.append(min(max(dom_use / tot_use, 1e-6), 1 - 1e-6))
        if HFCE_tot_v[j] <= 1e-9:
            gamma_cons.append(gamma_prod[-1])
        else:
            gamma_cons.append(min(max(HFCE_dom_v[j] / HFCE_tot_v[j], 1e-6), 1 - 1e-6))

    # Rho (constant)
    rho_val = 0.2
    rho = [[rho_val for _ in range(n)] for __ in range(n)]
    rho_cons = [rho_val for _ in range(n)]

    # A, imports, exports
    A = [1.0 for _ in range(n)]
    import_cost = [1.0 for _ in range(n)]
    exports = [max(x, 0.0) for x in EXPO_v]

    # Assemble block
    block = {
        'alpha_ij': alpha,
        'gamma_ij': [[gamma_prod[j] for j in range(n)] for _ in range(n)],  # broadcast per i
        'rho_ij': rho,
        'beta_j': beta,
        'A_i': A,
        'Export_i': exports,
        'gamma_cj': gamma_cons,
        'rho_cj': rho_cons,
        'import_cost': import_cost,
    }
    # Tradable set (agg6): {Agriculture, Mining, Manufacturing}
    tradable = list(range(n))
    if n == 6:
        tradable = [0, 1, 2]  # services & utilities non-tradable
    return {'params': block, 'n': n, 'tradable': tradable}


def main():
    ap = argparse.ArgumentParser(description="OECD USA IO -> eco_simu params (single-country block)")
    ap.add_argument('--oecd-dir', type=Path, required=True, help='Path to OECD美国投入产出数据 directory')
    ap.add_argument('--year', type=int, default=2019)
    ap.add_argument('--aggregate-6', action='store_true', help='Aggregate 50 sectors to 6 groups')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    oecd = load_oecd_us_tables(args.oecd_dir, args.year)
    built = build_country_params_from_oecd(oecd, aggregate6=args.aggregate_6)

    out = {
        'USA': built['params'],
        'tradable_sectors': built['tradable'],
        # Convenience 2-country dict for quick tests only
        'H': built['params'],
        'F': built['params'],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} (n={built['n']}, tradable={built['tradable']})")


if __name__ == '__main__':
    main()

