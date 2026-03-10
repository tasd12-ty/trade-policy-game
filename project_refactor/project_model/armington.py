"""Armington/CES 数学函数。"""

from __future__ import annotations

import numpy as np

from .utils import EPS, clamp_positive


def _sigma_from_rho(rho: np.ndarray | float) -> np.ndarray:
    r = np.asarray(rho, dtype=float)
    close = np.isclose(1.0 - r, 0.0, atol=1e-10)
    sigma = np.empty_like(r, dtype=float)
    sigma[close] = 1.0
    sigma[~close] = 1.0 / (1.0 - r[~close])
    return sigma


def armington_share(gamma: np.ndarray | float, p_dom: np.ndarray | float, p_for: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray:
    """Armington 份额函数。

    公式：
        sigma = 1 / (1-rho)
        theta = (g^sigma p_d^(1-sigma)) / (g^sigma p_d^(1-sigma) + (1-g)^sigma p_f^(1-sigma))
    """

    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    p_d = clamp_positive(p_dom)
    p_f = clamp_positive(p_for)
    r = np.asarray(rho, dtype=float)
    sigma = _sigma_from_rho(r)

    # rho -> 1 时对应完全替代极限，回到“选低价”规则。
    near_perfect = np.abs(1.0 - r) < 1e-4
    if np.isscalar(r) or np.ndim(r) == 0:
        if bool(near_perfect):
            if float(p_d) < float(p_f):
                return np.asarray(1.0)
            if float(p_d) > float(p_f):
                return np.asarray(0.0)
            return np.asarray(0.5)
    w_d = (g ** sigma) * (p_d ** (1.0 - sigma))
    w_f = ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    share = w_d / np.clip(w_d + w_f, EPS, None)

    # 数组情况的极限修正。
    if np.any(near_perfect):
        share = np.asarray(share, dtype=float)
        low = p_d < p_f
        high = p_d > p_f
        eq = ~(low | high)
        share = np.where(near_perfect & low, 1.0, share)
        share = np.where(near_perfect & high, 0.0, share)
        share = np.where(near_perfect & eq, 0.5, share)

    return np.clip(share, 1e-8, 1.0 - 1e-8)


def armington_price(gamma: np.ndarray | float, p_dom: np.ndarray | float, p_for: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray:
    """Armington 对偶价格函数。"""

    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    p_d = clamp_positive(p_dom)
    p_f = clamp_positive(p_for)
    r = np.asarray(rho, dtype=float)
    sigma = _sigma_from_rho(r)

    near_perfect = np.abs(1.0 - r) < 1e-4
    if np.isscalar(r) or np.ndim(r) == 0:
        if bool(near_perfect):
            return np.asarray(min(float(p_d), float(p_f)))

    near_sigma1 = np.abs(sigma - 1.0) < 1e-8
    if np.isscalar(sigma) or np.ndim(sigma) == 0:
        if bool(near_sigma1):
            return np.asarray(np.exp(g * np.log(p_d) + (1.0 - g) * np.log(p_f)))

    inner = (g ** sigma) * (p_d ** (1.0 - sigma)) + ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    out = np.clip(inner, EPS, None) ** (1.0 / (1.0 - sigma))

    if np.any(near_sigma1):
        geom = np.exp(g * np.log(p_d) + (1.0 - g) * np.log(p_f))
        out = np.where(near_sigma1, geom, out)
    if np.any(near_perfect):
        out = np.where(near_perfect, np.minimum(p_d, p_f), out)

    return clamp_positive(out)


def armington_quantity(gamma: float, x_dom: float, x_for: float, alpha: float, rho: float) -> float:
    """生产端有效投入的 Armington 合成量。"""

    if alpha <= 0.0:
        return 1.0
    g = float(np.clip(gamma, 1e-8, 1.0 - 1e-8))
    xd = float(max(x_dom, EPS))
    xf = float(max(x_for, EPS))
    r = float(rho)

    if abs(r) < 1e-10:
        return float(np.exp(alpha * (g * np.log(xd) + (1.0 - g) * np.log(xf))))

    comp = g * (xd ** r) + (1.0 - g) * (xf ** r)
    return float(max(comp, EPS) ** (alpha / r))
