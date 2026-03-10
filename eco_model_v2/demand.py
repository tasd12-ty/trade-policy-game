"""需求方程与动态调整量。

公式对应：
- 动态需求调整量（log-差分形式）：
  - compute_delta_x_dom:  eq 11, 12  国内中间品需求调整
  - compute_delta_x_imp:  eq 13      进口中间品需求调整
  - compute_delta_x_fac:  eq 14      要素需求调整
  - compute_delta_c_dom:  eq 16, 17  国内消费需求调整
  - compute_delta_c_imp:  eq 18      进口消费需求调整

- 静态均衡需求（水平值形式）：
  - static_intermediate_demand_dom: eq 31, 32  国内中间品需求
  - static_intermediate_demand_imp: eq 33      进口中间品需求
  - static_consumption_demand_dom:  eq 35, 36  国内消费需求
  - static_consumption_demand_imp:  eq 34      进口消费需求

依赖：utils.py
不依赖任何求解器或状态结构。
"""

from __future__ import annotations

import numpy as np

from .utils import EPS, safe_log, clamp_positive


# ============================================================
# 动态需求调整量（eq 11-18）
# ============================================================

def compute_delta_x_dom(
    alpha: np.ndarray,
    lambdas: np.ndarray,
    output: np.ndarray,
    theta: np.ndarray,
    price: np.ndarray,
    X_dom: np.ndarray,
    Ml: int,
    M_factors: int = 0,
) -> np.ndarray:
    """国内中间品 + 要素需求调整量 (eq 11, 12, 14)。

    eq 11 (非贸易品 j=1..Ml):
        Δx_{D,ij} = ln α_{ij} + ln λ_i − ln P_j + ln Y_i − ln X_{t;ij}

    eq 12 (可贸易品国内部分 j=Ml+1..Nl):
        Δx^I_{D,ij} = ln α_{ij} + ln λ_i + ln θ_{t;ij}
                     − ln P^I_j + ln Y_i − ln X^I_{t;ij}

    eq 14 (要素 j=Nl+1..Nl+M):
        Δx_{D,i,Nl+j} = ln α_{i,Nl+j} + ln λ_i
                        − ln P_{Nl+j} + ln Y_i − ln L_{t;ij}

    所有方程统一为：
        Δ = ln(α) + ln(λ) + ln(Y) + [ln(θ) if tradable] − ln(P) − ln(X_dom)

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        lambdas:   (Nl,)       边际成本
        output:    (Nl,)       当期产出
        theta:     (Nl, Nl)    Armington 国内份额
        price:     (Nl+M,)     产品+要素价格
        X_dom:     (Nl, Nl+M)  当期国内使用量
        Ml:        int          非贸易部门数
        M_factors: int          要素种类数

    返回：
        delta: (Nl, Nl+M) log-差分调整量
    """
    Nl = alpha.shape[0]
    total_cols = Nl + M_factors

    log_alpha = np.where(alpha > 0, safe_log(alpha), 0.0)
    log_lambda = safe_log(lambdas)[:, np.newaxis]         # (Nl, 1)
    log_Y = safe_log(output)[:, np.newaxis]               # (Nl, 1)
    log_price = safe_log(price)[np.newaxis, :total_cols]  # (1, Nl+M)
    log_X_dom = safe_log(X_dom)                           # (Nl, Nl+M)

    # 基础 delta（适用于非贸易品和要素）
    delta = log_alpha + log_lambda + log_Y - log_price - log_X_dom

    # 可贸易品国内部分：额外加 ln(θ)
    log_theta = safe_log(theta)  # (Nl, Nl)
    delta[:, Ml:Nl] += log_theta[:, Ml:Nl]

    # α≤0 的位置归零
    delta = np.where(alpha[:, :total_cols] > 0, delta, 0.0)

    return delta


def compute_delta_x_imp(
    alpha: np.ndarray,
    lambdas: np.ndarray,
    output: np.ndarray,
    theta: np.ndarray,
    imp_price: np.ndarray,
    X_imp: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """进口中间品需求调整量 (eq 13)。

    Δx^O_{D,ij} = ln α_{ij} + ln λ_i + ln(1−θ_{t;ij})
                 − ln P^O_j + ln Y_i − ln X^O_{t;ij},  j=Ml+1..Nl

    参数：
        alpha:     (Nl, Nl)   产出弹性（仅产品列）
        lambdas:   (Nl,)      边际成本
        output:    (Nl,)      产出
        theta:     (Nl, Nl)   Armington 国内份额
        imp_price: (Nl,)      进口品到岸价
        X_imp:     (Nl, Nl)   进口中间品使用量
        Ml:        int         非贸易部门数

    返回：
        delta: (Nl, Nl) log-差分调整量（非贸易列为零）
    """
    Nl = alpha.shape[0]

    log_alpha = np.where(alpha > 0, safe_log(alpha), 0.0)
    log_lambda = safe_log(lambdas)[:, np.newaxis]
    log_Y = safe_log(output)[:, np.newaxis]
    log_one_minus_theta = safe_log(1.0 - theta)
    log_imp_price = safe_log(imp_price)[np.newaxis, :]
    log_X_imp = safe_log(X_imp)

    delta = (log_alpha + log_lambda + log_one_minus_theta
             + log_Y - log_imp_price - log_X_imp)

    # 非贸易品列 (j < Ml) 无进口，归零
    delta[:, :Ml] = 0.0
    # α≤0 归零
    delta = np.where(alpha > 0, delta, 0.0)

    return delta


def compute_delta_c_dom(
    beta: np.ndarray,
    theta_cons: np.ndarray,
    income: float,
    price: np.ndarray,
    C_dom: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """国内消费需求调整量 (eq 16, 17)。

    eq 16 (非贸易品 j=1..Ml):
        Δc_{D,j} = ln β_j + ln I − ln P_j − ln C_{t;j}

    eq 17 (可贸易品国内部分 j=Ml+1..Nl):
        Δc^I_{D,j} = ln β_j + ln I + ln θ_{t;cj} − ln P^I_j − ln C^I_{t;j}

    参数：
        beta:       (Nl,)  消费预算权重
        theta_cons: (Nl,)  消费端 Armington 国内份额
        income:     float  消费者收入
        price:      (Nl,) 或 (Nl+M,)  产品价格
        C_dom:      (Nl,)  当期国内消费
        Ml:         int     非贸易部门数

    返回：
        delta: (Nl,) log-差分调整量
    """
    Nl = len(beta)
    log_beta = safe_log(beta)
    log_I = float(safe_log(income))
    log_price = safe_log(price[:Nl])
    log_C_dom = safe_log(C_dom)

    # 基础（非贸易品）
    delta = log_beta + log_I - log_price - log_C_dom

    # 可贸易品：额外加 ln(θ_c)
    log_theta_c = safe_log(theta_cons)
    delta[Ml:] += log_theta_c[Ml:]

    return delta


def compute_delta_c_imp(
    beta: np.ndarray,
    theta_cons: np.ndarray,
    income: float,
    imp_price: np.ndarray,
    C_imp: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """进口消费需求调整量 (eq 18)。

    Δc^O_{D,j} = ln β_j + ln I + ln(1−θ_{t;cj})
                − ln P^O_j − ln C^O_{t;j},  j=Ml+1..Nl

    参数：
        beta:       (Nl,)  消费预算权重
        theta_cons: (Nl,)  消费端 Armington 国内份额
        income:     float  消费者收入
        imp_price:  (Nl,)  进口品到岸价
        C_imp:      (Nl,)  当期进口消费
        Ml:         int     非贸易部门数

    返回：
        delta: (Nl,) log-差分调整量（非贸易部门为零）
    """
    Nl = len(beta)
    log_beta = safe_log(beta)
    log_I = float(safe_log(income))
    log_one_minus_theta = safe_log(1.0 - theta_cons)
    log_imp_price = safe_log(imp_price)
    log_C_imp = safe_log(C_imp)

    delta = (log_beta + log_I + log_one_minus_theta
             - log_imp_price - log_C_imp)

    # 非贸易品无进口消费
    delta[:Ml] = 0.0

    return delta


# ============================================================
# 静态均衡需求（eq 31-36）
# ============================================================

def static_intermediate_demand_dom(
    alpha: np.ndarray,
    theta: np.ndarray,
    price: np.ndarray,
    output: np.ndarray,
    Ml: int,
    M_factors: int = 0,
) -> np.ndarray:
    """静态均衡国内中间品需求 (eq 31, 32)。

    eq 31 (非贸易品 j=1..Ml):
        P_j · X_{ij} = α_{ij} · P_i · Y_i
        → X_{ij} = α_{ij} · P_i · Y_i / P_j

    eq 32 (可贸易品国内部分 j=Ml+1..Nl):
        P^I_j · X^I_{ij} = α_{ij} · θ_{ij} · P_i · Y_i
        → X^I_{ij} = α_{ij} · θ_{ij} · P_i · Y_i / P^I_j

    要素部分 (j=Nl+1..Nl+M):
        X_{i,Nl+k} = α_{i,Nl+k} · P_i · Y_i / P_{Nl+k}

    参数：
        alpha:     (Nl, Nl+M)  产出弹性
        theta:     (Nl, Nl)    Armington 国内份额
        price:     (Nl+M,)     产品+要素价格
        output:    (Nl,)       部门产出
        Ml:        int          非贸易部门数
        M_factors: int          要素种类数

    返回：
        X_dom: (Nl, Nl+M) 国内中间品 + 要素使用量
    """
    Nl = alpha.shape[0]
    total_cols = Nl + M_factors
    p = clamp_positive(price[:total_cols])
    y = np.asarray(output, dtype=float)
    PY = (p[:Nl] * y)[:, np.newaxis]  # (Nl, 1) 各部门收入

    X_dom = np.zeros((Nl, total_cols), dtype=float)

    # 非贸易品 + 要素：X = α · PY_i / P_j
    for j in list(range(Ml)) + list(range(Nl, total_cols)):
        X_dom[:, j] = alpha[:, j] * PY[:, 0] / p[j]

    # 可贸易品国内部分：X = α · θ · PY_i / P_j
    for j in range(Ml, Nl):
        X_dom[:, j] = alpha[:, j] * theta[:, j] * PY[:, 0] / p[j]

    return clamp_positive(X_dom)


def static_intermediate_demand_imp(
    alpha: np.ndarray,
    theta: np.ndarray,
    imp_price: np.ndarray,
    price: np.ndarray,
    output: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """静态均衡进口中间品需求 (eq 33)。

    P^O_j · X^O_{ij} = α_{ij} · (1−θ_{ij}) · P_i · Y_i
    → X^O_{ij} = α_{ij} · (1−θ_{ij}) · P_i · Y_i / P^O_j

    参数：
        alpha:     (Nl, Nl)   产出弹性（产品列）
        theta:     (Nl, Nl)   Armington 国内份额
        imp_price: (Nl,)      进口品到岸价
        price:     (Nl+M,) 或 (Nl,)  产品价格
        output:    (Nl,)      部门产出
        Ml:        int         非贸易部门数

    返回：
        X_imp: (Nl, Nl) 进口中间品使用量（非贸易列为零）
    """
    Nl = alpha.shape[0]
    p = clamp_positive(price[:Nl])
    y = np.asarray(output, dtype=float)
    PY = (p * y)[:, np.newaxis]  # (Nl, 1)

    p_imp = clamp_positive(imp_price)

    X_imp = np.zeros((Nl, Nl), dtype=float)
    for j in range(Ml, Nl):
        X_imp[:, j] = alpha[:, j] * (1.0 - theta[:, j]) * PY[:, 0] / p_imp[j]

    # 非贸易品严格无进口
    X_imp[:, :Ml] = 0.0
    return np.maximum(X_imp, 0.0)


def static_consumption_demand_dom(
    beta: np.ndarray,
    theta_cons: np.ndarray,
    income: float,
    price: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """静态均衡国内消费需求 (eq 35, 36)。

    eq 36 (非贸易品 j=1..Ml):
        P_j · C_j = β_j · I  → C_j = β_j · I / P_j

    eq 35 (可贸易品国内部分 j=Ml+1..Nl):
        P^I_j · C^I_j = β_j · θ_{cj} · I  → C^I_j = β_j · θ_{cj} · I / P^I_j

    参数：
        beta:       (Nl,)  消费预算权重
        theta_cons: (Nl,)  消费端 Armington 份额
        income:     float  消费者收入
        price:      (Nl,) 或 (Nl+M,)  产品价格
        Ml:         int     非贸易部门数

    返回：
        C_dom: (Nl,) 国内消费
    """
    Nl = len(beta)
    p = clamp_positive(price[:Nl])
    b = np.asarray(beta, dtype=float)
    I = max(float(income), EPS)

    C_dom = np.empty(Nl, dtype=float)

    # 非贸易品
    C_dom[:Ml] = b[:Ml] * I / p[:Ml]

    # 可贸易品国内部分
    C_dom[Ml:] = b[Ml:] * theta_cons[Ml:] * I / p[Ml:]

    return clamp_positive(C_dom)


def static_consumption_demand_imp(
    beta: np.ndarray,
    theta_cons: np.ndarray,
    income: float,
    imp_price: np.ndarray,
    Ml: int,
) -> np.ndarray:
    """静态均衡进口消费需求 (eq 34)。

    P^O_j · C^O_j = β_j · (1−θ_{cj}) · I
    → C^O_j = β_j · (1−θ_{cj}) · I / P^O_j

    参数：
        beta:       (Nl,)  消费预算权重
        theta_cons: (Nl,)  消费端 Armington 份额
        income:     float  消费者收入
        imp_price:  (Nl,)  进口品到岸价
        Ml:         int     非贸易部门数

    返回：
        C_imp: (Nl,) 进口消费（非贸易部门为零）
    """
    Nl = len(beta)
    p_imp = clamp_positive(imp_price)
    b = np.asarray(beta, dtype=float)
    I = max(float(income), EPS)

    C_imp = np.zeros(Nl, dtype=float)
    C_imp[Ml:] = b[Ml:] * (1.0 - theta_cons[Ml:]) * I / p_imp[Ml:]

    # 非贸易品严格无进口消费
    C_imp[:Ml] = 0.0
    return np.maximum(C_imp, 0.0)
