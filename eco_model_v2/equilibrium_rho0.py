"""ρ≡0 时的均衡解析解。

当所有 Armington 替代弹性 ρ≡0 时，CES 退化为 Cobb-Douglas，
模型存在闭合矩阵解。

公式对应：
- eq 37: α̃_{ij} 有效投入份额
- eq 38: PY = B · P·Export  产出值方程
- eq 39: ln P = (I−α̃)^{-1} · (...)  对数价格系统
- eq 40: 要素价格
- eq 41: 收入

依赖：factors.py, utils.py
"""

from __future__ import annotations

import numpy as np

from .factors import factor_prices_rho0, income_rho0
from .utils import EPS, safe_log, clamp_positive


def build_alpha_tilde(
    alpha: np.ndarray,
    theta: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """构造有效投入份额矩阵 α̃ (eq 37)。

    α̃_{ij} = {
        α_{ij} · θ_{ij},   j = 1, ..., Ml        （非贸易品：全额计入）
        α_{ij},             j = Ml+1, ..., Nl+M   （可贸易+要素：按 α 计入）
    }

    注：非贸易品 j < Ml 的 θ_{ij} = 1（无进口替代），
    所以 α̃_{ij} = α_{ij}·1 = α_{ij}。
    但为保持公式完整性，仍使用 θ 乘以。

    实际上 eq 37 的定义是：
    - j = Ml+1,...,Nl, Nl+1,...,Nl+M: α̃_{ij} = α_{ij}
    - j = 1,...,Ml: α̃_{ij} = α_{ij}·θ_{ij}

    但当 ρ=0 时 θ=γ（常数），所以 α̃ 是常数矩阵。

    参数：
        alpha: (Nl, Nl+M)  产出弹性
        theta: (Nl, Nl)    Armington 份额（ρ=0 时 = γ）
        Ml:    int          非贸易部门数

    返回：
        alpha_tilde: (Nl, Nl) 有效投入份额（仅产品部门列）
    """
    Nl = alpha.shape[0]
    total_cols = alpha.shape[1]

    # 取产品部门列 (Nl, Nl)
    at = np.array(alpha[:, :Nl], copy=True, dtype=float)

    # 非贸易品列乘以 θ（强制 θ[:, :Ml]=1，避免校准偏移）
    safe_theta = np.array(theta, copy=True, dtype=float)
    safe_theta[:, :Ml] = 1.0
    at[:, :Ml] *= safe_theta[:, :Ml]

    return at


def solve_prices_rho0(
    alpha: np.ndarray,
    alpha_tilde: np.ndarray,
    theta: np.ndarray,
    log_imp_price: np.ndarray,
    log_A: np.ndarray,
    Ml: int,
    M_factors: int = 0,
    log_factor_price: np.ndarray | None = None,
) -> np.ndarray:
    """ρ≡0 对数价格系统 (eq 39)。

    ln P = (I − α̃)^{-1} · (α^O · ln P^O + Σ_k α_{Nl+k} · ln P_{Nl+k} − ln A)

    其中：
    - α^O_{ij} = α_{ij}·(1−θ_{ij}), j=Ml+1..Nl（进口份额弹性）
    - P^O 为进口品价格
    - P_{Nl+k} 为要素价格

    参数：
        alpha:            (Nl, Nl+M) 完整弹性矩阵
        alpha_tilde:      (Nl, Nl)   有效份额矩阵
        theta:            (Nl, Nl)   Armington 份额（ρ=0 时 = γ）
        log_imp_price:    (Nl,)      进口品对数价格
        log_A:            (Nl,)      ln(TFP)
        Ml:               int         非贸易部门数
        M_factors:        int         要素种类数
        log_factor_price: (M,) 或 None  要素对数价格

    返回：
        log_price: (Nl,) 国内产品对数价格
    """
    Nl = alpha_tilde.shape[0]

    # 构造 α^O 矩阵 (Nl, Nl)：进口份额弹性
    alpha_O = np.zeros((Nl, Nl), dtype=float)
    for j in range(Ml, Nl):
        alpha_O[:, j] = alpha[:, j] * (1.0 - theta[:, j])

    # 右端向量
    rhs = -np.asarray(log_A, dtype=float).copy()

    # 加进口价格贡献
    for j in range(Ml, Nl):
        rhs += alpha_O[:, j] * float(log_imp_price[j])

    # 加要素价格贡献 (eq 39 完整形式)
    if log_factor_price is not None and M_factors > 0:
        lfp = np.asarray(log_factor_price, dtype=float)
        for k in range(M_factors):
            rhs += alpha[:, Nl + k] * float(lfp[k])

    # 求解 (I − α̃) · p = rhs  → p = (I − α̃)^{-1} · rhs
    I_minus_at = np.eye(Nl, dtype=float) - alpha_tilde
    try:
        log_price = np.linalg.solve(I_minus_at, rhs)
    except np.linalg.LinAlgError:
        log_price = np.linalg.lstsq(I_minus_at, rhs, rcond=None)[0]

    return log_price


def solve_output_rho0(
    alpha_tilde: np.ndarray,
    beta: np.ndarray,
    price: np.ndarray,
    exports: np.ndarray,
    Nl: int,
    M_factors: int = 0,
) -> np.ndarray:
    """ρ≡0 产出值方程 (eq 38)。

    PY = B · P·Export

    其中 B = (I − α̃^T − Σ_k β · α̃^T_{Nl+k})^{-1}

    简化形式（忽略消费对进口的回流）：
    B = (I − α̃^T)^{-1}

    然后 PY_j = Σ_i B_{ji} · P_i · Export_i

    参数：
        alpha_tilde: (Nl, Nl) 有效份额矩阵
        beta:        (Nl,)    消费预算权重
        price:       (Nl,) 或 (Nl+M,)  产品价格
        exports:     (Nl,) 或 (Nl+M,)  出口量
        Nl:          int      部门数
        M_factors:   int      要素种类数

    返回：
        PY: (Nl,) 各部门产出值 P_i · Y_i
    """
    p = np.asarray(price[:Nl], dtype=float)
    exp = np.asarray(exports[:Nl], dtype=float)
    P_Export = p * exp

    # B = (I − α̃^T)^{-1}
    I_minus_at_T = np.eye(Nl, dtype=float) - alpha_tilde.T
    try:
        B = np.linalg.inv(I_minus_at_T)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(I_minus_at_T)

    PY = B @ P_Export
    return clamp_positive(PY)


def solve_equilibrium_rho0(
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    exports: np.ndarray,
    imp_price: np.ndarray,
    L: np.ndarray,
    Ml: int,
    M_factors: int,
) -> dict:
    """ρ≡0 完整均衡解析解。

    步骤：
    1. θ = γ（ρ=0 时 Armington 份额等于权重参数）
    2. 构造 α̃ (eq 37)
    3. 求解对数价格 (eq 39)
    4. 求解产出值 PY (eq 38)
    5. 计算要素价格 (eq 40)
    6. 计算收入 (eq 41)

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        gamma:     (Nl, Nl)    Armington 权重
        beta:      (Nl,)       消费预算权重
        A:         (Nl,)       TFP
        exports:   (Nl+M,)     出口量
        imp_price: (Nl,)       进口品到岸价
        L:         (M,)        要素禀赋
        Ml:        int          非贸易部门数
        M_factors: int          要素种类数

    返回：
        dict 包含 price, output, PY, factor_prices, income, alpha_tilde
    """
    Nl = alpha.shape[0]

    # 1. ρ=0 时 θ = γ
    theta = np.clip(np.asarray(gamma, dtype=float), 1e-8, 1.0 - 1e-8)

    # 2. 构造 α̃
    alpha_tilde = build_alpha_tilde(alpha, theta, Ml)

    log_imp = safe_log(imp_price)
    log_A = safe_log(A)

    # 3. 迭代求解产品价格与要素价格的耦合系统
    #    步骤：初始化要素价格 → 解产品价格(eq 39) → 名义锚归一
    #          → 解 PY(eq 38) → 更新要素价格(eq 40) → 迭代至收敛
    #    名义锚：P[0] = 1（每轮迭代后对产品+要素价格统一缩放）
    log_factor_p = np.zeros(M_factors, dtype=float)  # 初始 ln(P_fac) = 0

    max_iter = 100
    tol = 1e-10
    damping = 0.3
    for _it in range(max_iter):
        log_price = solve_prices_rho0(
            alpha, alpha_tilde, theta, log_imp, log_A, Ml, M_factors,
            log_factor_price=log_factor_p if M_factors > 0 else None,
        )
        # 名义锚：令 P[0] = 1 → 对所有价格做相对调整
        anchor_shift = log_price[0]
        log_price = log_price - anchor_shift

        price_product = clamp_positive(np.exp(log_price))
        PY = solve_output_rho0(alpha_tilde, beta, price_product, exports, Nl, M_factors)

        if M_factors > 0:
            new_factor_p = factor_prices_rho0(alpha, PY, L, Nl, M_factors)
            new_log_fp = safe_log(new_factor_p)
            delta = float(np.max(np.abs(new_log_fp - log_factor_p)))
            log_factor_p = damping * log_factor_p + (1 - damping) * new_log_fp
            if delta < tol:
                break
        else:
            break

    price_product = clamp_positive(np.exp(log_price))
    PY = solve_output_rho0(alpha_tilde, beta, price_product, exports, Nl, M_factors)

    # 4. 产出量 Y = PY / P
    output = PY / clamp_positive(price_product)

    # 5. 要素价格 (eq 40)
    factor_p = factor_prices_rho0(alpha, PY, L, Nl, M_factors)

    # 6. 收入 (eq 41)
    income = income_rho0(alpha, PY, Nl, M_factors)

    # 7. 组装完整价格向量
    full_price = np.concatenate([price_product, factor_p])

    return {
        "price": full_price,
        "output": output,
        "PY": PY,
        "factor_prices": factor_p,
        "income": income,
        "alpha_tilde": alpha_tilde,
        "theta": theta,
    }
