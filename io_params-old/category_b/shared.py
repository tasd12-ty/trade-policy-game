"""B 类参数计算的通用工具函数。"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def safe_positive(value: float, default: float = 1.0) -> float:
    """保证标量为正值，避免后续计算出现除零或对数奇异。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v) or v <= 0.0:
        return float(default)
    return v


def sanitize_rho(value: float) -> float:
    """清洗 rho，避免 rho=1 导致 CES 指数奇异。"""

    v = float(value)
    if not np.isfinite(v):
        return 0.2
    if abs(v - 1.0) < 1e-8:
        v = 0.95
    return float(np.clip(v, -5.0, 0.95))

