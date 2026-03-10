"""C 类进口价格路径 p_t^O 计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .external_inputs import ExternalCInputs
from .shared import EPS, safe_positive


def _build_p_t_o_multiplier_matrix(
    sectors: list[str],
    external: ExternalCInputs,
    periods: int,
) -> pd.DataFrame:
    """构造逐期逐部门进口价格乘子矩阵。"""

    mat = pd.DataFrame(
        safe_positive(external.p_t_o_multiplier_default, 1.0),
        index=pd.Index(range(periods), name="period_t"),
        columns=sectors,
        dtype=float,
    )

    # 先应用“按期统一乘子”。
    for t, m in external.p_t_o_multiplier_scalar_by_period.items():
        if 0 <= int(t) < periods:
            mat.loc[int(t), :] = safe_positive(m, external.p_t_o_multiplier_default)

    # 再应用“按期按部门乘子”（优先级更高）。
    for t, mapping in external.p_t_o_multiplier_by_period_sector.items():
        if not (0 <= int(t) < periods):
            continue
        for s, m in mapping.items():
            if s not in mat.columns:
                continue
            mat.loc[int(t), s] = safe_positive(m, external.p_t_o_multiplier_default)
    return mat


def compute_p_t_o(
    p_t_o_base_j: pd.Series,
    external: ExternalCInputs,
    periods: int | None = None,
) -> pd.DataFrame:
    """计算 `p_t^O`。

    公式：`p_t^O[j] = p_base^O[j] * multiplier_t[j]`。
    """

    sectors = list(p_t_o_base_j.index)
    t_len = int(periods if periods is not None else external.horizon_periods)
    t_len = max(t_len, 1)

    base = p_t_o_base_j.reindex(sectors).fillna(external.p_t_o_base_default).astype(float)
    base = base.apply(lambda x: safe_positive(x, external.p_t_o_base_default))
    mult = _build_p_t_o_multiplier_matrix(sectors=sectors, external=external, periods=t_len)

    arr = mult.to_numpy(dtype=float) * base.to_numpy(dtype=float).reshape(1, -1)
    p_t_o = pd.DataFrame(arr, index=mult.index, columns=sectors)
    p_t_o = p_t_o.clip(lower=EPS)
    p_t_o = p_t_o.replace([np.inf, -np.inf], np.nan).fillna(EPS)
    p_t_o.columns.name = "sector_j"
    return p_t_o

