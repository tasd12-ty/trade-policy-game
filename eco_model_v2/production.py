"""生产函数与边际成本。

公式对应：
- compute_output:        eq 4  生产函数 Y_i
- compute_cost:          eq 5  成本函数 M_i
- compute_marginal_cost: eq 10 边际成本 λ_i

依赖：utils.py, armington.py
不依赖任何求解器或状态结构。
"""

from __future__ import annotations

import numpy as np

from .utils import EPS, safe_log, clamp_positive
from .armington import armington_quantity, armington_price


# ---- eq 4: 生产函数 ----

def compute_output(
    A: np.ndarray,
    alpha: np.ndarray,
    X_dom: np.ndarray,
    X_imp: np.ndarray,
    gamma: np.ndarray,
    rho: np.ndarray,
    Ml: int,
    M_factors: int = 0,
) -> np.ndarray:
    """生产函数 (eq 4)。

    Y_i = A_i · ∏_{j=1}^{Ml} X_{ij}^{α_{ij}}
              · ∏_{j=Ml+1}^{Nl} [γ·X_d^ρ + (1-γ)·X_f^ρ]^{α/ρ}
              · ∏_{k=1}^{M} L_{ik}^{α_{i,Nl+k}}

    参数：
        A:     (Nl,)       全要素生产率
        alpha: (Nl, Nl+M)  产出弹性矩阵（后 M 列为要素）
        X_dom: (Nl, Nl+M)  国内中间品 + 要素使用量
        X_imp: (Nl, Nl)    进口中间品（仅可贸易列非零）
        gamma: (Nl, Nl)    Armington 国内权重
        rho:   (Nl, Nl)    Armington 形状参数
        Ml:    int          非贸易部门数
        M_factors: int      要素种类数

    返回：
        Y: (Nl,) 各部门产出
    """
    Nl = alpha.shape[0]
    total_cols = Nl + M_factors
    Y = np.empty(Nl, dtype=float)

    for i in range(Nl):
        prod = max(float(A[i]), EPS)

        for j in range(total_cols):
            a = float(alpha[i, j])
            if a <= 0.0:
                continue

            if j < Ml:
                # 非贸易中间品：直接 Cobb-Douglas
                prod *= max(float(X_dom[i, j]), EPS) ** a
            elif j < Nl:
                # 可贸易中间品：Armington CES 合成
                qty = armington_quantity(
                    gamma=float(gamma[i, j]),
                    x_dom=float(X_dom[i, j]),
                    x_for=float(X_imp[i, j]),
                    alpha=a,
                    rho=float(rho[i, j]),
                )
                prod *= max(float(qty), EPS)
            else:
                # 要素投入（劳动/资本）：Cobb-Douglas
                prod *= max(float(X_dom[i, j]), EPS) ** a

        Y[i] = max(prod, EPS)

    return Y


# ---- eq 5: 成本函数 ----

def compute_cost(
    alpha: np.ndarray,
    X_dom: np.ndarray,
    X_imp: np.ndarray,
    price: np.ndarray,
    imp_price: np.ndarray,
    Ml: int,
    M_factors: int = 0,
) -> np.ndarray:
    """成本函数 (eq 5)。

    M_i = Σ_{j=1}^{Ml} P_j · X_{ij}
        + Σ_{j=Ml+1}^{Nl} (P_j · X^I_{ij} + P^O_j · X^O_{ij})
        + Σ_{k=1}^{M} P_{Nl+k} · L_{ik}

    参数：
        alpha:     (Nl, Nl+M)  产出弹性（仅用于判断是否使用）
        X_dom:     (Nl, Nl+M)  国内中间品 + 要素使用量
        X_imp:     (Nl, Nl)    进口中间品
        price:     (Nl+M,)     产品价格 + 要素价格
        imp_price: (Nl,)       进口品到岸价
        Ml:        int          非贸易部门数
        M_factors: int          要素种类数

    返回：
        M: (Nl,) 各部门总成本
    """
    Nl = alpha.shape[0]
    total_cols = Nl + M_factors
    cost = np.zeros(Nl, dtype=float)

    for i in range(Nl):
        for j in range(total_cols):
            if alpha[i, j] <= 0.0:
                continue
            if j < Ml:
                # 非贸易品：国内价格 × 使用量
                cost[i] += float(price[j]) * float(X_dom[i, j])
            elif j < Nl:
                # 可贸易品：国内部分 + 进口部分
                cost[i] += (float(price[j]) * float(X_dom[i, j])
                            + float(imp_price[j]) * float(X_imp[i, j]))
            else:
                # 要素：要素价格 × 要素使用量
                cost[i] += float(price[j]) * float(X_dom[i, j])

    return cost


# ---- eq 10: 边际成本 ----

def compute_marginal_cost(
    A: np.ndarray,
    alpha: np.ndarray,
    price: np.ndarray,
    imp_price: np.ndarray,
    gamma: np.ndarray,
    rho: np.ndarray,
    Ml: int,
    M_factors: int = 0,
) -> np.ndarray:
    """边际成本——对偶价格形式 (eq 10)。

    完整形式：
        ln λ_i = −ln A_i − Σ_j α_{ij}·ln(α_{ij}) + Σ_j α_{ij}·ln P*_j

    其中 P*_j 为：
    - 非贸易品 j < Ml:     P*_j = P_j
    - 可贸易品 Ml ≤ j < Nl: P*_j = Armington 对偶价格
    - 要素 j ≥ Nl:         P*_j = P_j（要素价格）

    −Σ α·ln(α) 项为 Cobb-Douglas 的 CES 约化常数，保证
    当所有 P*_j = 1 时 λ_i = 1/A_i · ∏ α^{-α}。

    参数：
        A:         (Nl,)       TFP
        alpha:     (Nl, Nl+M)  产出弹性
        price:     (Nl+M,)     产品 + 要素价格
        imp_price: (Nl,)       进口品到岸价
        gamma:     (Nl, Nl)    Armington 权重
        rho:       (Nl, Nl)    Armington 形状参数
        Ml:        int          非贸易部门数
        M_factors: int          要素种类数

    返回：
        lambdas: (Nl,) 各部门边际成本
    """
    Nl = alpha.shape[0]
    total_cols = Nl + M_factors
    lambdas = np.empty(Nl, dtype=float)

    for i in range(Nl):
        log_cost = -float(safe_log(A[i]))

        for j in range(total_cols):
            a = float(alpha[i, j])
            if a <= 0.0:
                continue

            # eq 10 常数项：−α_{ij}·ln(α_{ij})
            log_cost -= a * float(safe_log(a))

            if j < Ml:
                # 非贸易品
                log_cost += a * float(safe_log(price[j]))
            elif j < Nl:
                # 可贸易品：使用 Armington 对偶价格
                p_star = float(armington_price(
                    gamma=float(gamma[i, j]),
                    p_dom=float(price[j]),
                    p_for=float(imp_price[j]),
                    rho=float(rho[i, j]),
                ))
                log_cost += a * float(safe_log(p_star))
            else:
                # 要素
                log_cost += a * float(safe_log(price[j]))

        lambdas[i] = float(np.exp(log_cost))

    return clamp_positive(lambdas)


