"""C 类外需路径 Export_t 计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .external_inputs import ExternalCInputs
from .shared import safe_non_negative


def _build_export_multiplier_matrix(
    sectors: list[str],
    external: ExternalCInputs,
    periods: int,
) -> pd.DataFrame:
    """构造逐期逐部门出口乘子矩阵。"""

    mat = pd.DataFrame(
        safe_non_negative(external.export_multiplier_default, 1.0),
        index=pd.Index(range(periods), name="period_t"),
        columns=sectors,
        dtype=float,
    )

    # 先应用“按期统一乘子”。
    for t, m in external.export_multiplier_scalar_by_period.items():
        if 0 <= int(t) < periods:
            mat.loc[int(t), :] = safe_non_negative(m, external.export_multiplier_default)

    # 再应用“按期按部门乘子”（优先级更高）。
    for t, mapping in external.export_multiplier_by_period_sector.items():
        if not (0 <= int(t) < periods):
            continue
        for s, m in mapping.items():
            if s not in mat.columns:
                continue
            mat.loc[int(t), s] = safe_non_negative(m, external.export_multiplier_default)
    return mat


def compute_export_t(
    export_base_j: pd.Series,
    external: ExternalCInputs,
    periods: int | None = None,
) -> pd.DataFrame:
    """计算 `Export_t`。

    公式：`Export_t[j] = Export_base_j[j] * multiplier_t[j]`。
    """

    sectors = list(export_base_j.index)
    t_len = int(periods if periods is not None else external.horizon_periods)
    t_len = max(t_len, 1)

    base = export_base_j.reindex(sectors).fillna(external.export_base_default).astype(float)
    base = base.apply(lambda x: safe_non_negative(x, external.export_base_default))
    mult = _build_export_multiplier_matrix(sectors=sectors, external=external, periods=t_len)

    arr = mult.to_numpy(dtype=float) * base.to_numpy(dtype=float).reshape(1, -1)
    export_t = pd.DataFrame(arr, index=mult.index, columns=sectors)
    export_t = export_t.clip(lower=0.0)
    export_t = export_t.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    export_t.columns.name = "sector_j"
    return export_t

