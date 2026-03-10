"""C 类参数计算通用工具。"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


def safe_positive(value: float, default: float = 1.0) -> float:
    """保证标量为正值。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v) or v <= 0.0:
        return float(default)
    return float(v)


def safe_non_negative(value: float, default: float = 0.0) -> float:
    """保证标量为非负值。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v) or v < 0.0:
        return float(default)
    return float(v)


def safe_probability(value: float, default: float = 1.0) -> float:
    """将份额参数清洗到 [0,1]。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(np.clip(default, 0.0, 1.0))
    if not np.isfinite(v):
        return float(np.clip(default, 0.0, 1.0))
    return float(np.clip(v, 0.0, 1.0))


def safe_rho(value: float, default: float = 1.0) -> float:
    """清洗 rho，避免 rho=1 导致 CES 奇异。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    if abs(v - 1.0) < 1e-8:
        v = 0.95
    return float(np.clip(v, -5.0, 0.95))


def build_sector_series(
    sectors: Iterable[str],
    *,
    default_value: float,
    overrides: Dict[str, float],
    non_negative: bool = False,
    positive: bool = False,
    probability: bool = False,
    rho_value: bool = False,
    name: str,
    index_name: str,
) -> pd.Series:
    """按部门代码构造序列，并应用覆盖值。"""

    out = pd.Series(index=list(sectors), dtype=float, name=name)
    for s in out.index:
        raw = overrides.get(s, default_value)
        if probability:
            out.loc[s] = safe_probability(raw, default_value)
        elif rho_value:
            out.loc[s] = safe_rho(raw, default_value)
        elif positive:
            out.loc[s] = safe_positive(raw, default_value)
        elif non_negative:
            out.loc[s] = safe_non_negative(raw, default_value)
        else:
            out.loc[s] = float(raw)
    out.index.name = index_name
    return out

