"""静态均衡求解器。

公式对应：
- eq 28: 零利润条件 λ_i = P_i
- eq 29: 要素市场出清 Σ α_{i,Nl+k}·P_i·Y_i = P_{Nl+k}·L_k
- eq 30: 国际收支平衡
- eq 31-36: 需求方程（从价格/产出导出数量）

求解策略（精简变量法）：
- 未知变量仅为：价格 (Nl+M) + 产出 (Nl)，共 2*(2Nl+M)+2 个变量
- 需求量从价格和产出通过一阶条件直接计算（非独立变量）
- 使用 log 参数化强制非负
- 主求解器：scipy.optimize.least_squares
- 回退：固定点阻尼迭代

依赖：production.py, factors.py, demand.py, armington.py, utils.py, types.py
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .armington import armington_share_from_prices
from .production import compute_output, compute_marginal_cost
from .factors import compute_income, factor_clearing_residual
from .demand import (
    static_intermediate_demand_dom,
    static_intermediate_demand_imp,
    static_consumption_demand_dom,
    static_consumption_demand_imp,
)
from .types import (
    CountryParams,
    TwoCountryParams,
    CountryBlock,
    StaticEquilibriumResult,
)
from .utils import EPS, relative_error, clamp_positive


# ============================================================
# 精简变量求解：仅以 (price, output) 为未知量
# ============================================================

def _pack_vars(price_h: np.ndarray, output_h: np.ndarray,
               price_f: np.ndarray, output_f: np.ndarray) -> np.ndarray:
    """打包两国 (price, output) 为一维 log 向量。"""
    return np.log(np.clip(
        np.concatenate([price_h, output_h, price_f, output_f]), EPS, None
    ))


def _unpack_vars(log_vec: np.ndarray, Nl: int, M: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """解包 log 向量为两国 (price, output)。"""
    v = np.exp(log_vec)
    sz = Nl + M + Nl  # price (Nl+M) + output (Nl)
    price_h = v[:Nl + M]
    output_h = v[Nl + M:sz]
    price_f = v[sz:sz + Nl + M]
    output_f = v[sz + Nl + M:]
    return price_h, output_h, price_f, output_f


def _compute_country_quantities(
    params: CountryParams,
    price: np.ndarray,
    output: np.ndarray,
    partner_price: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """从 (price, output) 导出全部需求量。

    返回：X_dom, X_imp, C_dom, C_imp, imp_price, income
    """
    Nl = params.Nl
    Ml = int(params.Ml)
    M = int(params.M_factors)
    imp_price = np.asarray(params.import_cost, dtype=float) * partner_price[:Nl]

    # Armington 份额（价格端）
    theta = np.ones((Nl, Nl), dtype=float)
    for i in range(Nl):
        for j in range(Ml, Nl):
            theta[i, j] = float(armington_share_from_prices(
                params.gamma[i, j], price[j], imp_price[j], params.rho[i, j],
            ))

    theta_c = np.ones(Nl, dtype=float)
    for j in range(Ml, Nl):
        theta_c[j] = float(armington_share_from_prices(
            params.gamma_cons[j], price[j], imp_price[j], params.rho_cons[j],
        ))

    income = compute_income(price, params.L, Nl)

    X_dom = static_intermediate_demand_dom(params.alpha, theta, price, output, Ml, M)
    X_imp = static_intermediate_demand_imp(params.alpha[:, :Nl], theta, imp_price, price, output, Ml)
    C_dom = static_consumption_demand_dom(params.beta, theta_c, income, price, Ml)
    C_imp = static_consumption_demand_imp(params.beta, theta_c, income, imp_price, Ml)

    return X_dom, X_imp, C_dom, C_imp, imp_price, income


def _build_residuals(
    log_vec: np.ndarray,
    params: TwoCountryParams,
) -> np.ndarray:
    """构造两国均衡残差（精简版）。

    残差组成（每国）：
    1. 零利润 (eq 28): P_i = λ_i(price)          — Nl 个
    2. 要素市场出清 (eq 29)                        — M 个
    3. 商品市场出清 (eq 19=0 的静态对应，全部 Nl 个部门)
    4. 贸易收支 (eq 30)                            — 1 个
    全局：
    5. 名义锚: w_H = 1, w_F = 1 (要素价格锚)     — 2 个
       注：锚在要素价格（而非商品价格）避免与零利润冲突。

    总残差数 = 2×(Nl + M + Nl + 1) + 2（过识别最小二乘）
    未知量   = 2×(Nl+M + Nl)
    """
    Nl = params.Nl
    M = params.M

    price_h, output_h, price_f, output_f = _unpack_vars(log_vec, Nl, M)

    # 导出需求量
    X_dom_h, X_imp_h, C_dom_h, C_imp_h, imp_h, _ = _compute_country_quantities(
        params.home, price_h, output_h, price_f)
    X_dom_f, X_imp_f, C_dom_f, C_imp_f, imp_f, _ = _compute_country_quantities(
        params.foreign, price_f, output_f, price_h)

    res = []

    for cp, price, output, X_dom, X_imp, C_dom, C_imp, imp_price in [
            (params.home, price_h, output_h, X_dom_h, X_imp_h, C_dom_h, C_imp_h, imp_h),
            (params.foreign, price_f, output_f, X_dom_f, X_imp_f, C_dom_f, C_imp_f, imp_f),
    ]:
        # 1. 零利润 (eq 28): P_i = λ_i
        lambdas = compute_marginal_cost(
            cp.A, cp.alpha, price, imp_price,
            cp.gamma, cp.rho, int(cp.Ml), int(cp.M_factors),
        )
        for i in range(Nl):
            res.append(float(relative_error(price[i], lambdas[i])))

        # 2. 要素市场出清 (eq 29)
        fac_res = factor_clearing_residual(
            cp.alpha, price, output, cp.L, Nl, M,
        )
        for k in range(M):
            supply_val = max(float(price[Nl + k]) * float(cp.L[k]), 1.0)
            res.append(float(fac_res[k]) / supply_val)

        # 3. 商品市场出清（按 PDF：Export 为外生量，不依赖对手内生进口）
        exports_qty = cp.exports[:Nl]
        for j in range(Nl):
            total_demand = float(X_dom[:, j].sum() + C_dom[j] + exports_qty[j])
            res.append(float(relative_error(output[j], total_demand)))

        # 4. 贸易收支 (eq 30)
        exports_value = float(np.dot(price[:Nl], cp.exports[:Nl]))
        imports_value = float(np.dot(imp_price, X_imp.sum(axis=0) + C_imp))
        TRADE_BALANCE_WEIGHT = 5.0
        res.append(float(relative_error(exports_value, imports_value)) * TRADE_BALANCE_WEIGHT)

    # 名义锚：要素价格 w=1（避免与零利润条件冲突）
    if M > 0:
        res.append(float(relative_error(price_h[Nl], 1.0)))
        res.append(float(relative_error(price_f[Nl], 1.0)))
    else:
        # 无要素时回退到商品价格锚
        res.append(float(relative_error(price_h[0], 1.0)))
        res.append(float(relative_error(price_f[0], 1.0)))

    return np.asarray(res, dtype=float)


# ============================================================
# 初值
# ============================================================

def _initial_guess(params: TwoCountryParams) -> np.ndarray:
    """构造初值 log 向量——使用 ρ=0 解析解作为暖启动。

    关键：将 ρ=0 解归一化到 w=1（要素价格锚），使初始猜测
    与残差函数的名义锚一致。同时调整产出水平满足要素市场约束。
    """
    from .equilibrium_rho0 import solve_equilibrium_rho0

    Nl, M = params.Nl, params.M

    guesses = []
    for i, cp in enumerate((params.home, params.foreign)):
        partner = params.foreign if i == 0 else params.home
        imp_p = np.asarray(cp.import_cost, dtype=float)

        try:
            gc = np.asarray(cp.gamma_cons, dtype=float) if cp.gamma_cons is not None else None
            rho0 = solve_equilibrium_rho0(
                alpha=np.asarray(cp.alpha, dtype=float),
                gamma=np.asarray(cp.gamma, dtype=float),
                beta=np.asarray(cp.beta, dtype=float),
                A=np.asarray(cp.A, dtype=float),
                exports=np.asarray(cp.exports, dtype=float),
                imp_price=imp_p,
                L=np.asarray(cp.L, dtype=float),
                Ml=int(cp.Ml),
                M_factors=int(cp.M_factors),
                gamma_cons=gc,
            )
            price = clamp_positive(rho0["price"])
            output = clamp_positive(rho0["output"])

            # 归一化: 将价格缩放到 w=1（要素价格锚）
            if M > 0:
                w = price[Nl]
                if w > EPS:
                    price = price / w

            # 缩放产出满足要素市场约束: Σ α_factor * P * Y = w * L
            alpha_f = np.asarray(cp.alpha[:, Nl:], dtype=float)
            factor_demand = (alpha_f * price[:Nl, None] * output[:, None]).sum()
            factor_supply = float(np.dot(price[Nl:], cp.L))
            if factor_demand > EPS:
                scale = factor_supply / factor_demand
                output = output * scale

        except Exception:
            price = np.ones(Nl + M, dtype=float)
            output = np.maximum(np.asarray(cp.A, dtype=float), 1.0)

        guesses.extend([price, output])

    return np.log(np.clip(np.concatenate(guesses), EPS, None))


# ============================================================
# 结果构造
# ============================================================

def _build_country_block(
    params: CountryParams,
    price: np.ndarray,
    output: np.ndarray,
    partner_price: np.ndarray,
) -> CountryBlock:
    """从求解结果构造 CountryBlock。"""
    X_dom, X_imp, C_dom, C_imp, imp_price, income = _compute_country_quantities(
        params, price, output, partner_price,
    )
    return CountryBlock(
        X_dom=X_dom, X_imp=X_imp, C_dom=C_dom, C_imp=C_imp,
        price=price, imp_price=imp_price, output=output, income=income,
    )


# ============================================================
# 主求解入口
# ============================================================

def solve_static_equilibrium(
    params: TwoCountryParams,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
) -> StaticEquilibriumResult:
    """求解两国静态均衡。

    精简变量法：仅以 (price, output) 为未知量（各国 Nl+M+Nl 个）。
    需求量从一阶条件直接导出，大幅减少变量维度。

    参数：
        params: 两国参数
        max_iterations: 最大函数求值次数
        tolerance: 收敛容差

    返回：
        StaticEquilibriumResult
    """
    Nl, M = params.Nl, params.M
    log_guess = _initial_guess(params)

    def residual_fn(v: np.ndarray) -> np.ndarray:
        return _build_residuals(v, params)

    try:
        from scipy.optimize import least_squares

        result = least_squares(
            residual_fn,
            log_guess,
            method='lm',
            ftol=float(tolerance),
            xtol=float(tolerance),
            gtol=float(tolerance),
            max_nfev=int(max_iterations),
            verbose=0,
        )
        log_sol = result.x
        converged = bool(result.success)
        iterations = int(result.nfev)
        final_residual = float(np.linalg.norm(residual_fn(log_sol)))
        solver_message = str(result.message)

    except ImportError:
        log_sol, converged, iterations, final_residual, solver_message = (
            _solve_fallback(params, log_guess, max_iterations, tolerance)
        )

    price_h, output_h, price_f, output_f = _unpack_vars(log_sol, Nl, M)

    home_block = _build_country_block(params.home, price_h, output_h, price_f)
    foreign_block = _build_country_block(params.foreign, price_f, output_f, price_h)

    return StaticEquilibriumResult(
        home=home_block,
        foreign=foreign_block,
        converged=converged,
        iterations=iterations,
        final_residual=final_residual,
        solver_message=solver_message,
    )


# ============================================================
# 回退求解器
# ============================================================

def _solve_fallback(
    params: TwoCountryParams,
    log_guess: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> Tuple[np.ndarray, bool, int, float, str]:
    """无 scipy 时的固定点迭代回退。"""
    Nl, M = params.Nl, params.M
    x = np.array(log_guess, copy=True)

    def residual_fn(v: np.ndarray) -> np.ndarray:
        return _build_residuals(v, params)

    best_res = np.inf
    best_x = x.copy()

    for it in range(1, max_iterations + 1):
        price_h, output_h, price_f, output_f = _unpack_vars(x, Nl, M)

        # 更新每国
        for cp, price, output, partner_price in [
            (params.home, price_h, output_h, price_f),
            (params.foreign, price_f, output_f, price_h),
        ]:
            imp_price = np.asarray(cp.import_cost, dtype=float) * partner_price[:Nl]
            lambdas = compute_marginal_cost(
                cp.A, cp.alpha, price, imp_price,
                cp.gamma, cp.rho, int(cp.Ml), int(cp.M_factors),
            )
            # 阻尼价格更新
            price[:Nl] = np.clip(0.7 * price[:Nl] + 0.3 * lambdas, EPS, None)

            # 更新产出
            X_dom, X_imp, C_dom, C_imp, _, _ = _compute_country_quantities(
                cp, price, output, partner_price)
            Y_prod = compute_output(
                cp.A, cp.alpha, X_dom, X_imp,
                cp.gamma, cp.rho, int(cp.Ml), int(cp.M_factors),
            )
            output[:] = np.clip(0.7 * output + 0.3 * Y_prod, EPS, None)

        # 名义锚：要素价格 w=1
        if M > 0:
            price_h[Nl] = 1.0
            price_f[Nl] = 1.0
        else:
            price_h[0] = 1.0
            price_f[0] = 1.0

        x = _pack_vars(price_h, output_h, price_f, output_f)
        res_norm = float(np.linalg.norm(residual_fn(x)))

        if res_norm < best_res:
            best_res = res_norm
            best_x = x.copy()

        if res_norm < tolerance:
            return x, True, it, res_norm, "fixed_point_converged"

    return best_x, False, max_iterations, best_res, "fixed_point_max_iter"
