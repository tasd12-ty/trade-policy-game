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


# ---- CES 价格归一化常数 ----

def ces_price_norm(
    gamma: float | np.ndarray,
    rho: float | np.ndarray,
    ic: float | np.ndarray = 1.0,
) -> np.ndarray:
    """CES 价格归一化常数 ln(P*/P_d)。

    P* = P_d · exp(c)，其中 c = ces_price_norm(γ, ρ, ic)。

    一般形式：
        c = ln([γ^σ + (1-γ)^σ · ic^{1-σ}]^{1/(1-σ)})

    CD 极限 (ρ→0, σ→1)：
        c = (1-γ)·ln(ic) − γ·ln γ − (1-γ)·ln(1-γ)
    """
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    r = np.asarray(rho, dtype=float)
    ic_v = np.maximum(np.asarray(ic, dtype=float), EPS)
    sigma = _sigma_from_rho(r)

    near_cd = np.abs(sigma - 1.0) < 1e-8

    # CD 极限
    cd_c = (1.0 - g) * safe_log(ic_v) - g * safe_log(g) - (1.0 - g) * safe_log(1.0 - g)

    # 一般 CES
    safe_sigma = np.where(near_cd, 2.0, sigma)  # 避免 sigma=1 除零
    safe_one_minus = np.where(near_cd, -1.0, 1.0 - safe_sigma)
    term = g ** safe_sigma + (1.0 - g) ** safe_sigma * ic_v ** safe_one_minus
    gen_c = safe_log(np.maximum(term, EPS)) / safe_one_minus

    return np.where(near_cd, cd_c, gen_c)


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

    # 一般情形——使用对数空间 sigmoid 避免 p^(1-σ) 下溢
    # ln(w_d) = σ·ln(γ) + (1-σ)·ln(p_d)
    # ln(w_f) = σ·ln(1-γ) + (1-σ)·ln(p_f)
    # θ = 1 / (1 + exp(ln_wf - ln_wd))  = sigmoid(ln_wd - ln_wf)
    ln_wd = sigma * safe_log(g) + (1.0 - sigma) * safe_log(p_d)
    ln_wf = sigma * safe_log(1.0 - g) + (1.0 - sigma) * safe_log(p_f)
    diff = ln_wd - ln_wf
    # 数值稳定的 sigmoid：clamp 避免 exp 溢出
    diff_clamped = np.clip(diff, -500.0, 500.0)
    share = 1.0 / (1.0 + np.exp(-diff_clamped))

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
    - σ→1: P_d^γ · P_f^{1-γ} / (γ^γ · (1-γ)^{1-γ})  (CD 对偶成本)
    """
    g = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)
    p_d = clamp_positive(p_dom)
    p_f = clamp_positive(p_for)
    r = np.asarray(rho, dtype=float)
    sigma = _sigma_from_rho(r)

    near_perfect = np.abs(1.0 - r) < 1e-4
    near_cd = np.abs(sigma - 1.0) < 1e-8

    # 一般情形——对数空间计算避免 p^(1-σ) 下溢
    # P* = [γ^σ·p_d^{1-σ} + (1-γ)^σ·p_f^{1-σ}]^{1/(1-σ)}
    # ln(P*) = (1/(1-σ)) · ln[exp(ln_wd) + exp(ln_wf)]  使用 log-sum-exp
    ln_wd = sigma * safe_log(g) + (1.0 - sigma) * safe_log(p_d)
    ln_wf = sigma * safe_log(1.0 - g) + (1.0 - sigma) * safe_log(p_f)
    # log-sum-exp: ln(e^a + e^b) = max(a,b) + ln(1 + exp(-|a-b|))
    ln_max = np.maximum(ln_wd, ln_wf)
    ln_sum = ln_max + np.log1p(np.exp(-np.abs(ln_wd - ln_wf)))
    # 1/(1-sigma) 可以为负或很大；直接用 exp
    safe_one_minus_sigma = np.where(near_cd, 1.0, 1.0 - sigma)
    out = np.exp(ln_sum / safe_one_minus_sigma)

    # σ=1 CD 对偶修正：P* = P_d^γ · P_f^{1-γ} / (γ^γ · (1-γ)^{1-γ})
    # L'Hôpital 极限包含 −γ·ln γ − (1-γ)·ln(1-γ) 熵项
    cd_val = np.exp(
        g * safe_log(p_d) + (1.0 - g) * safe_log(p_f)
        - g * safe_log(g) - (1.0 - g) * safe_log(1.0 - g)
    )
    out = np.where(near_cd, cd_val, out)

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
