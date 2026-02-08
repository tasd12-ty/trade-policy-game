"""要素市场方程。

公式对应：
- compute_income:              eq 7   消费者收入 I = Σ P_{Nl+k} · L_k
- compute_factor_demand:       eq 8   要素总需求 L_k^D = Σ_i L_{ik}
- factor_clearing_residual:    eq 29  要素市场出清残差
- factor_prices_rho0:          eq 40  要素价格（ρ≡0 解析解）
- income_rho0:                 eq 41  收入（ρ≡0 解析解）

依赖：utils.py
不依赖任何求解器或状态结构。
"""

from __future__ import annotations

import numpy as np

from .utils import EPS, clamp_positive


# ---- eq 7: 消费者收入 ----

def compute_income(
    price: np.ndarray,
    L: np.ndarray,
    Nl: int,
) -> float:
    """消费者收入 (eq 7)。

    I = Σ_{k=1}^{M} P_{Nl+k} · L_k

    收入 = 要素价格 × 要素禀赋的总和。
    要素不跨国流动，每国收入仅取决于本国要素。

    参数：
        price: (Nl+M,)  产品+要素价格，后 M 项为要素价格
        L:     (M,)     要素禀赋（劳动、资本等固定量）
        Nl:    int       部门数

    返回：
        income: float 消费者总收入
    """
    factor_prices = np.asarray(price[Nl:], dtype=float)
    endowments = np.asarray(L, dtype=float)
    return float(np.dot(factor_prices, endowments))


# ---- eq 7 替代形式：增加值法收入 ----

def compute_income_value_added(
    price: np.ndarray,
    output: np.ndarray,
    value_added_share: np.ndarray,
) -> float:
    """增加值法收入（不使用要素价格时的替代计算）。

    I = Σ_i P_i · Y_i · v_i

    其中 v_i = 1 − Σ_j α_{ij} 为增加值份额。
    在要素市场出清时，此值与 eq 7 一致。

    参数：
        price:             (Nl,) 或 (Nl+M,)  产品价格
        output:            (Nl,)              部门产出
        value_added_share: (Nl,)              增加值份额 v_i

    返回：
        income: float
    """
    Nl = len(output)
    p = np.asarray(price[:Nl], dtype=float)
    y = np.asarray(output, dtype=float)
    va = np.asarray(value_added_share, dtype=float)
    return float(np.dot(p * y, va))


# ---- eq 8: 要素总需求 ----

def compute_factor_demand(
    X_dom: np.ndarray,
    Nl: int,
    M_factors: int,
) -> np.ndarray:
    """要素总需求 (eq 8)。

    L_k^D = Σ_{i=1}^{Nl} L_{ik}

    其中 L_{ik} = X_dom[i, Nl+k] 为部门 i 对要素 k 的使用量。

    参数：
        X_dom:     (Nl, Nl+M)  国内中间品 + 要素使用量
        Nl:        int          部门数
        M_factors: int          要素种类数

    返回：
        L_D: (M,) 各要素总需求
    """
    # X_dom 的后 M 列为要素使用量
    factor_usage = np.asarray(X_dom[:, Nl:Nl + M_factors], dtype=float)
    return factor_usage.sum(axis=0)


# ---- eq 29: 要素市场出清 ----

def factor_clearing_residual(
    alpha: np.ndarray,
    price: np.ndarray,
    output: np.ndarray,
    L: np.ndarray,
    Nl: int,
    M_factors: int,
) -> np.ndarray:
    """要素市场出清残差 (eq 29)。

    均衡条件：Σ_i α_{i,Nl+k} · P_i · Y_i = P_{Nl+k} · L_k

    残差 = Σ_i α_{i,Nl+k}·P_i·Y_i − P_{Nl+k}·L_k

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        price:     (Nl+M,)     产品+要素价格
        output:    (Nl,)       部门产出
        L:         (M,)        要素禀赋
        Nl:        int          部门数
        M_factors: int          要素种类数

    返回：
        residual: (M,) 各要素市场出清残差
    """
    p = np.asarray(price, dtype=float)
    y = np.asarray(output, dtype=float)
    endow = np.asarray(L, dtype=float)

    # P_i · Y_i 向量
    PY = p[:Nl] * y

    residual = np.empty(M_factors, dtype=float)
    for k in range(M_factors):
        # 要素需求侧：Σ_i α_{i,Nl+k} · P_i · Y_i
        demand_value = float(np.dot(alpha[:, Nl + k], PY))
        # 要素供给侧：P_{Nl+k} · L_k
        supply_value = float(p[Nl + k]) * float(endow[k])
        residual[k] = demand_value - supply_value

    return residual


# ---- eq 40: 要素价格（ρ≡0 解析解） ----

def factor_prices_rho0(
    alpha: np.ndarray,
    PY: np.ndarray,
    L: np.ndarray,
    Nl: int,
    M_factors: int,
) -> np.ndarray:
    """ρ≡0 时的要素价格 (eq 40)。

    P_{Nl+k} = (1/L_k) · Σ_i α_{i,Nl+k} · P_i · Y_i
             = (α_{Nl+k}^T · PY) / L_k

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        PY:        (Nl,)       P_i · Y_i 向量
        L:         (M,)        要素禀赋
        Nl:        int          部门数
        M_factors: int          要素种类数

    返回：
        factor_prices: (M,) 要素价格
    """
    PY_arr = np.asarray(PY, dtype=float)
    endow = clamp_positive(np.asarray(L, dtype=float))

    factor_p = np.empty(M_factors, dtype=float)
    for k in range(M_factors):
        # α_{Nl+k}^T · PY
        factor_p[k] = float(np.dot(alpha[:, Nl + k], PY_arr)) / float(endow[k])

    return clamp_positive(factor_p)


# ---- eq 41: 收入（ρ≡0 解析解） ----

def income_rho0(
    alpha: np.ndarray,
    PY: np.ndarray,
    Nl: int,
    M_factors: int,
) -> float:
    """ρ≡0 时的收入 (eq 41)。

    I = Σ_{k=1}^{M} P_{Nl+k} · L_k
      = 1_M^T · (α_{Nl+1}^T, ..., α_{Nl+M}^T)^T · PY
      = Σ_{k=1}^{M} (α_{Nl+k}^T · PY)

    注：因 P_{Nl+k}·L_k = α_{Nl+k}^T·PY（eq 40），
    收入直接等于所有要素列乘 PY 的总和。

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        PY:        (Nl,)       P_i · Y_i 向量
        Nl:        int          部门数
        M_factors: int          要素种类数

    返回：
        income: float
    """
    PY_arr = np.asarray(PY, dtype=float)
    total = 0.0
    for k in range(M_factors):
        total += float(np.dot(alpha[:, Nl + k], PY_arr))
    return max(total, EPS)
