"""通用数值工具。

本模块无外部依赖（仅 numpy），提供全局使用的数值安全函数。
"""

from __future__ import annotations

import numpy as np

# ---- 全局常量 ----

EPS = 1e-9  # 正数下界，防止 log(0) 或除零


# ---- 安全数学函数 ----

def safe_log(x: np.ndarray | float) -> np.ndarray:
    """安全对数：log(max(x, EPS))，避免非正值导致 NaN。"""
    return np.log(np.clip(np.asarray(x, dtype=float), EPS, None))


def safe_exp(x: np.ndarray | float, max_abs: float = 20.0) -> np.ndarray:
    """安全指数：截断极端值后取 exp，防止上溢。"""
    return np.exp(np.clip(np.asarray(x, dtype=float), -max_abs, max_abs))


def clamp_positive(x: np.ndarray | float, min_value: float = EPS) -> np.ndarray:
    """截断到正区间 [min_value, +∞)。"""
    return np.clip(np.asarray(x, dtype=float), min_value, None)


def tanh_damping(delta: np.ndarray | float, cap: float = 3.0) -> np.ndarray:
    """tanh 阻尼：限制单期变动幅度在 [-cap, cap] 内。

    公式：cap * tanh(delta / cap)
    当 delta 很小时近似线性，大时被截断。
    """
    d = np.asarray(delta, dtype=float)
    return cap * np.tanh(d / cap)


def relative_error(actual: np.ndarray | float,
                   expected: np.ndarray | float) -> np.ndarray:
    """相对误差缩放：(actual - expected) / max(|actual|, |expected|, 1)。

    避免量纲和尺度影响，用于构造稳健的残差。
    """
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(e)), 1.0)
    return (a - e) / scale
