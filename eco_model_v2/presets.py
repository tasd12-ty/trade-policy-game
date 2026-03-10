"""参数预设与构造工具。

提供对称/非对称两国参数的便捷构造函数，
以及从旧版 project_refactor 格式的兼容转换。

依赖：types.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .types import CountryParams, TwoCountryParams


def make_symmetric_params(
    Nl: int = 5,
    Ml: int = 2,
    M_factors: int = 1,
    *,
    alpha_diag: float = 0.15,
    alpha_off: float = 0.05,
    alpha_factor: float | None = None,
    gamma_tradable: float = 0.6,
    rho_tradable: float = 0.3,
    beta: np.ndarray | None = None,
    A: np.ndarray | None = None,
    L: np.ndarray | None = None,
    export_base: float = 0.5,
    import_cost_base: float = 1.1,
    gamma_cons: float = 0.6,
    rho_cons: float = 0.3,
) -> TwoCountryParams:
    """构造对称两国参数。

    参数：
        Nl:             部门数
        Ml:             非贸易部门数 (前 Ml 个部门不可贸易)
        M_factors:      要素种类数
        alpha_diag:     对角线产出弹性
        alpha_off:      非对角产出弹性
        alpha_factor:   要素投入弹性（None 则自动补齐使行和=1）
        gamma_tradable: 可贸易品 Armington 国内权重
        rho_tradable:   可贸易品 Armington 形状参数
        beta:           消费预算权重 (Nl,)，默认均匀
        A:              TFP (Nl,)，默认全 1
        L:              要素禀赋 (M,)，默认全 10
        export_base:    基准出口量（标量，应用于可贸易部门）
        import_cost_base: 进口成本乘子
        gamma_cons:     消费端 Armington 国内权重
        rho_cons:       消费端 Armington 形状参数

    返回：
        TwoCountryParams（两国完全对称）
    """
    M = M_factors

    # alpha (Nl, Nl+M)
    alpha = np.full((Nl, Nl + M), alpha_off, dtype=float)
    np.fill_diagonal(alpha[:, :Nl], alpha_diag)

    # 要素弹性列
    if alpha_factor is not None:
        alpha[:, Nl:] = alpha_factor
    else:
        # 自动补齐使行和 = 1
        row_sum_prod = alpha[:, :Nl].sum(axis=1)
        remaining = np.maximum(1.0 - row_sum_prod, 0.01)
        if M > 0:
            alpha[:, Nl:] = remaining[:, np.newaxis] / M

    # gamma (Nl, Nl)
    gamma = np.ones((Nl, Nl), dtype=float)
    gamma[:, Ml:] = gamma_tradable

    # rho (Nl, Nl)
    rho = np.zeros((Nl, Nl), dtype=float)
    rho[:, Ml:] = rho_tradable

    # beta
    if beta is None:
        beta_arr = np.ones(Nl, dtype=float) / Nl
    else:
        beta_arr = np.asarray(beta, dtype=float)

    # A
    if A is None:
        A_arr = np.ones(Nl, dtype=float)
    else:
        A_arr = np.asarray(A, dtype=float)

    # L
    if L is None:
        L_arr = np.full(M, 10.0, dtype=float)
    else:
        L_arr = np.asarray(L, dtype=float)

    # exports (Nl+M,)
    exports = np.zeros(Nl + M, dtype=float)
    exports[Ml:Nl] = export_base

    # import_cost (Nl,)
    import_cost = np.full(Nl, import_cost_base, dtype=float)
    import_cost[:Ml] = 1.0  # 非贸易品无进口成本

    # gamma_cons, rho_cons
    gc = np.ones(Nl, dtype=float)
    gc[Ml:] = gamma_cons
    rc = np.zeros(Nl, dtype=float)
    rc[Ml:] = rho_cons

    cp = CountryParams(
        alpha=alpha, gamma=gamma, rho=rho,
        beta=beta_arr, A=A_arr, exports=exports,
        gamma_cons=gc, rho_cons=rc,
        import_cost=import_cost, L=L_arr,
        Ml=Ml, M_factors=M,
    )

    # 深拷贝构造对称的 foreign
    foreign = CountryParams(
        alpha=alpha.copy(), gamma=gamma.copy(), rho=rho.copy(),
        beta=beta_arr.copy(), A=A_arr.copy(), exports=exports.copy(),
        gamma_cons=gc.copy(), rho_cons=rc.copy(),
        import_cost=import_cost.copy(), L=L_arr.copy(),
        Ml=Ml, M_factors=M,
    )

    return TwoCountryParams(home=cp, foreign=foreign)


def apply_tariff(
    params: CountryParams,
    tariff_rates: Dict[int, float],
    *,
    base_import_cost: np.ndarray | None = None,
) -> CountryParams:
    """对指定部门施加关税，返回新参数。

    关税通过修改 import_cost 实现（税率“水平”语义）：
    import_cost'[j] = base_import_cost[j] × (1 + tariff_rate[j])

    若未提供 base_import_cost，则默认使用 params.import_cost 作为基准。

    参数：
        params:       原始参数
        tariff_rates: {部门索引: 关税率}，如 {3: 0.2} 表示部门 3 加 20% 关税
        base_import_cost: (Nl,) 关税基准进口成本（可选）
    """
    base_ic = (
        np.asarray(base_import_cost, dtype=float)
        if base_import_cost is not None
        else np.asarray(params.import_cost, dtype=float)
    )
    if base_ic.shape != np.asarray(params.import_cost, dtype=float).shape:
        raise ValueError("base_import_cost 形状必须与 params.import_cost 一致")

    new_ic = np.array(params.import_cost, copy=True, dtype=float)
    for sector, rate in tariff_rates.items():
        if 0 <= sector < params.Nl:
            new_ic[sector] = float(base_ic[sector]) * (1.0 + max(float(rate), 0.0))

    return CountryParams(
        alpha=params.alpha, gamma=params.gamma, rho=params.rho,
        beta=params.beta, A=params.A, exports=params.exports,
        gamma_cons=params.gamma_cons, rho_cons=params.rho_cons,
        import_cost=new_ic, L=params.L,
        Ml=params.Ml, M_factors=params.M_factors,
    )


def apply_quota(
    params: CountryParams,
    quota_multipliers: Dict[int, float],
) -> CountryParams:
    """对指定部门施加出口配额，返回新参数。

    配额通过修改 exports 实现：
    exports'[j] = exports[j] × quota_mult[j]

    参数：
        params:            原始参数
        quota_multipliers: {部门索引: 配额乘子}，如 {3: 0.5} 表示部门 3 出口减半
    """
    new_exp = np.array(params.exports, copy=True, dtype=float)
    for sector, mult in quota_multipliers.items():
        if 0 <= sector < params.Nl:
            new_exp[sector] *= max(float(mult), 0.0)

    return CountryParams(
        alpha=params.alpha, gamma=params.gamma, rho=params.rho,
        beta=params.beta, A=params.A, exports=new_exp,
        gamma_cons=params.gamma_cons, rho_cons=params.rho_cons,
        import_cost=params.import_cost, L=params.L,
        Ml=params.Ml, M_factors=params.M_factors,
    )
