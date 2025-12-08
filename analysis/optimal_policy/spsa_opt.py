"""（分析用）黑箱优化器：SPSA 与有限差分 PGD。

用途：在“仅能取值—目标值”的黑箱设定下，做带边界的投影一阶搜索。
- SPSA：维度高时更高效（每步 O(d) 调用）；
- PGD（有限差分梯度）：维度低时更稳，便于验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np


ObjectiveFn = Callable[[np.ndarray], Tuple[float, dict]]


@dataclass
class SPSAConfig:
    iterations: int = 80
    a0: float = 0.2          # 初始步长
    c0: float = 0.1          # 初始扰动规模
    alpha: float = 0.602     # 步长衰减指数
    gamma: float = 0.101     # 扰动衰减指数
    seed: Optional[int] = 42


def project_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """投影到盒约束 [lo, hi]。"""
    return np.minimum(np.maximum(x, lo), hi)


def spsa(
    x0: np.ndarray,
    f: ObjectiveFn,
    lo: np.ndarray,
    hi: np.ndarray,
    cfg: SPSAConfig = SPSAConfig(),
) -> Tuple[np.ndarray, float, dict]:
    """SPSA 投影上升法（最大化目标）。

    - 每步把 x 投影回 [lo, hi]；
    - f(x) 返回 (目标值, 信息字典)。
    """
    rng = np.random.default_rng(cfg.seed)
    x = project_box(np.array(x0, float), lo, hi)
    best_x = x.copy()
    best_val, best_info = f(best_x)

    for k in range(1, cfg.iterations + 1):
        ak = cfg.a0 / (k ** cfg.alpha)
        ck = cfg.c0 / (k ** cfg.gamma)
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        x_plus = project_box(x + ck * delta, lo, hi)
        x_minus = project_box(x - ck * delta, lo, hi)
        f_plus, _ = f(x_plus)
        f_minus, _ = f(x_minus)
        ghat = (f_plus - f_minus) / (2.0 * ck * delta)
        x = project_box(x + ak * ghat, lo, hi)
        val, info = f(x)
        if val > best_val:
            best_val, best_x, best_info = val, x.copy(), info
    return best_x, float(best_val), best_info


def pgd_fd(
    x0: np.ndarray,
    f: ObjectiveFn,
    lo: np.ndarray,
    hi: np.ndarray,
    steps: int = 40,
    h: float = 1e-2,
    step_size: float = 0.1,
    momentum: float = 0.0,
) -> Tuple[np.ndarray, float, dict]:
    """有限差分 + 投影梯度上升（适合低维）。"""
    x = project_box(np.array(x0, float), lo, hi)
    v = np.zeros_like(x)
    best_x = x.copy()
    best_val, best_info = f(best_x)
    for t in range(steps):
        base_val, _ = f(x)
        g = np.zeros_like(x)
        for i in range(x.size):
            xh = x.copy()
            xh[i] = min(max(xh[i] + h, lo[i]), hi[i])
            fv, _ = f(xh)
            g[i] = (fv - base_val) / (xh[i] - x[i] if xh[i] != x[i] else h)
        v = momentum * v + (1.0 - momentum) * g
        x = project_box(x + step_size * v, lo, hi)
        val, info = f(x)
        if val > best_val:
            best_val, best_x, best_info = val, x.copy(), info
    return best_x, float(best_val), best_info


__all__ = [
    "SPSAConfig",
    "spsa",
    "pgd_fd",
]
