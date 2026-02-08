"""Armington/CES 纯数学函数。

本模块仅依赖 utils.py，不依赖任何求解器或状态。
提供价格端和用量端两种 Armington 份额计算，以及对偶价格和物量合成。

公式对应：
- armington_share_from_prices:  均衡求解时使用
- theta_from_quantities:        eq 9 / eq 15（动态更新时使用）
- armington_price:              对偶价格函数（边际成本计算用）
- armington_quantity:           物量合成（生产函数用）
"""

from __future__ import annotations

import numpy as np

from .utils import EPS, clamp_positive, safe_log


# ---- 替代弹性 ----

def _sigma_from_rho(rho: np.ndarray) -> np.ndarray:
    """ρ → σ = 1/(1-ρ)，σ=1 当 ρ≈0 时特殊处理。"""
    r = np.asarray(rho, dtype=float)
    close = np.isclose(1.0 - r, 0.0, atol=1e-10)
    sigma = np.where(close, 1.0, 1.0 / np.where(close, 1.0, 1.0 - r))
    return sigma


# ---- 价格端 Armington 份额 ----

def armington_share_from_prices(
    gamma: np.ndarray | float,
    p_dom: np.ndarray | float,
    p_for: np.ndarray | float,
    rho: np.ndarray | float,
) -> np.ndarray:
    """价格端 Armington 份额函数。

    公式：
        σ = 1/(1-ρ)
        θ = (γ^σ · p_d^{1-σ}) / (γ^σ · p_d^{1-σ} + (1-γ)^σ · p_f^{1-σ})

    返回 θ ∈ (0, 1)，表示国内产品在 CES 束中的支出份额。

    边界情形：
    - ρ→1 (σ→∞): 完全替代，选低价方
    - ρ→0 (σ→1): Cobb-Douglas，份额 = γ
    """
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    p_d = clamp_positive(p_dom)
    p_f = clamp_positive(p_for)
    r = np.asarray(rho, dtype=float)
    sigma = _sigma_from_rho(r)

    # 完全替代极限：ρ→1 时直接选低价
    near_perfect = np.abs(1.0 - r) < 1e-4
    # Cobb-Douglas 极限：σ=1 时 θ = γ
    near_cd = np.abs(sigma - 1.0) < 1e-8

    # 一般情形
    w_d = (g ** sigma) * (p_d ** (1.0 - sigma))
    w_f = ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    share = w_d / np.clip(w_d + w_f, EPS, None)

    # 边界修正
    share = np.where(near_cd, g, share)
    share = np.where(near_perfect & (p_d < p_f), 1.0, share)
    share = np.where(near_perfect & (p_d > p_f), 0.0, share)
    share = np.where(near_perfect & np.isclose(p_d, p_f), 0.5, share)

    return np.clip(np.asarray(share, dtype=float), 1e-8, 1.0 - 1e-8)


# ---- 用量端 Armington 份额（eq 9 / eq 15） ----

def theta_from_quantities(
    gamma: np.ndarray | float,
    x_dom: np.ndarray | float,
    x_for: np.ndarray | float,
    rho: np.ndarray | float,
) -> np.ndarray:
    """用量端 Armington 份额 —— 对应 eq 9 和 eq 15。

    实现采用标准 CES 形式（与 grad_op 一致）：
        θ_{t;ij} = γ_{ij} · (X^I_{ij})^{ρ}
                   / [γ_{ij} · (X^I_{ij})^{ρ} + (1-γ_{ij}) · (X^O_{ij})^{ρ}]

    注：PDF eq 9 字面记法为 (γ·X)^ρ，但标准 CES 应为 γ·X^ρ，
    后者具有正确的 Cobb-Douglas 极限（ρ→0 时 θ→γ）。
    若采用字面 (γ·X)^ρ 则 ρ→0 时 θ→0.5，经济含义不正确。

    含义：根据当前实际使用量（而非价格）计算国内外竞品比例。
    在动态更新中使用此函数（而非价格端份额）。

    当 |ρ| → 0 时退化为 θ = γ（Cobb-Douglas 极限）。
    """
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    xd = clamp_positive(x_dom)
    xf = clamp_positive(x_for)
    r = np.asarray(rho, dtype=float)

    # ρ≈0 极限：θ = γ
    near_zero = np.abs(r) < 1e-10

    # 一般情形
    dom_term = g * (xd ** r)
    for_term = (1.0 - g) * (xf ** r)
    denom = np.clip(dom_term + for_term, EPS, None)
    theta = dom_term / denom

    # ρ≈0 修正
    theta = np.where(near_zero, g, theta)

    return np.clip(np.asarray(theta, dtype=float), 1e-8, 1.0 - 1e-8)


# ---- Armington 对偶价格 ----

def armington_price(
    gamma: np.ndarray | float,
    p_dom: np.ndarray | float,
    p_for: np.ndarray | float,
    rho: np.ndarray | float,
) -> np.ndarray:
    """Armington 对偶价格（CES 单位成本函数）。

    公式：
        P* = [γ^σ · p_d^{1-σ} + (1-γ)^σ · p_f^{1-σ}]^{1/(1-σ)}

    边界：
    - σ→∞: min(p_d, p_f)
    - σ→1: exp(γ·ln(p_d) + (1-γ)·ln(p_f))  几何平均
    """
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    p_d = clamp_positive(p_dom)
    p_f = clamp_positive(p_for)
    r = np.asarray(rho, dtype=float)
    sigma = _sigma_from_rho(r)

    near_perfect = np.abs(1.0 - r) < 1e-4
    near_cd = np.abs(sigma - 1.0) < 1e-8

    # 一般情形
    inner = (g ** sigma) * (p_d ** (1.0 - sigma)) + \
            ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    out = np.clip(inner, EPS, None) ** (1.0 / (1.0 - sigma))

    # σ=1 几何平均修正
    geom = np.exp(g * safe_log(p_d) + (1.0 - g) * safe_log(p_f))
    out = np.where(near_cd, geom, out)

    # 完全替代修正
    out = np.where(near_perfect, np.minimum(p_d, p_f), out)

    return clamp_positive(out)


# ---- Armington 物量合成 ----

def armington_quantity(
    gamma: np.ndarray | float,
    x_dom: np.ndarray | float,
    x_for: np.ndarray | float,
    alpha: np.ndarray | float,
    rho: np.ndarray | float,
) -> np.ndarray:
    """Armington 物量合成（生产函数中的有效投入）。

    公式 (eq 4 中可贸易部分)：
        component = [γ · x_d^ρ + (1-γ) · x_f^ρ]^{α/ρ}

    当 α ≤ 0 时返回 1.0（该投入不使用）。
    当 ρ → 0 时退化为 Cobb-Douglas：exp(α · (γ·ln(x_d) + (1-γ)·ln(x_f)))。

    支持标量和数组输入。
    """
    a = np.asarray(alpha, dtype=float)
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    xd = clamp_positive(x_dom)
    xf = clamp_positive(x_for)
    r = np.asarray(rho, dtype=float)

    # α ≤ 0 的部分返回 1
    inactive = a <= 0.0
    near_zero = np.abs(r) < 1e-10

    # Cobb-Douglas 极限
    cd_val = np.exp(a * (g * safe_log(xd) + (1.0 - g) * safe_log(xf)))

    # 一般 CES
    # 使用安全除法防止 r=0 导致的除零
    safe_r = np.where(near_zero, 1.0, r)
    comp = g * (xd ** safe_r) + (1.0 - g) * (xf ** safe_r)
    ces_val = clamp_positive(comp) ** (a / safe_r)

    # 组合
    result = np.where(near_zero, cd_val, ces_val)
    result = np.where(inactive, 1.0, result)

    return clamp_positive(result)
