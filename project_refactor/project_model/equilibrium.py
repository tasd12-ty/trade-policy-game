"""静态均衡求解器（对应论文静态方程组）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .armington import armington_share
from .presets import from_raw_dict
from .production import compute_income, compute_marginal_cost, compute_output
from .types import CountryBlock, CountryParams, StaticEquilibriumResult, TwoCountryParams
from .utils import EPS, relative_error


StateMap = Dict[str, Dict[str, np.ndarray]]


@dataclass
class EquilibriumLayout:
    """均衡变量布局。

    为了控制维度，进口相关变量仅对可贸易部门保留压缩列：
    - X_imp: (n, n_tradable)
    - C_imp: (n_tradable,)
    """

    n: int
    tradable_idx: np.ndarray

    def __post_init__(self) -> None:
        self.tradable_idx = np.asarray(sorted(self.tradable_idx.tolist()), dtype=int)
        self.n_tradable = len(self.tradable_idx)
        self.tradable_mask = np.zeros(self.n, dtype=bool)
        self.tradable_mask[self.tradable_idx] = True
        self.non_tradable_idx = np.asarray([i for i in range(self.n) if not self.tradable_mask[i]], dtype=int)
        self._tradable_pos = {int(j): k for k, j in enumerate(self.tradable_idx)}

        self._fields: list[tuple[str, str, tuple[int, ...]]] = []
        for c in ("H", "F"):
            self._fields.extend(
                [
                    (c, "X_dom", (self.n, self.n)),
                    (c, "X_imp", (self.n, self.n_tradable)),
                    (c, "C_dom", (self.n,)),
                    (c, "C_imp", (self.n_tradable,)),
                    (c, "price", (self.n,)),
                ]
            )
        self._slices: Dict[tuple[str, str], tuple[int, int, tuple[int, ...]]] = {}
        cur = 0
        for c, f, shp in self._fields:
            size = int(np.prod(shp))
            self._slices[(c, f)] = (cur, cur + size, shp)
            cur += size
        self.total_size = cur

    def new_state(self, fill: float = EPS) -> StateMap:
        st: StateMap = {}
        for c in ("H", "F"):
            st[c] = {
                "X_dom": np.full((self.n, self.n), fill, dtype=float),
                "X_imp": np.full((self.n, self.n_tradable), fill, dtype=float),
                "C_dom": np.full((self.n,), fill, dtype=float),
                "C_imp": np.full((self.n_tradable,), fill, dtype=float),
                "price": np.ones((self.n,), dtype=float),
            }
        return st

    def pack(self, state: StateMap) -> np.ndarray:
        vec = np.empty(self.total_size, dtype=float)
        for (c, f), (s, e, _) in self._slices.items():
            vec[s:e] = np.asarray(state[c][f], dtype=float).reshape(-1)
        return vec

    def unpack(self, vector: np.ndarray) -> StateMap:
        st: StateMap = {"H": {}, "F": {}}
        for (c, f), (s, e, shp) in self._slices.items():
            st[c][f] = np.asarray(vector[s:e], dtype=float).reshape(shp)
        return st

    def expand_matrix(self, condensed: np.ndarray) -> np.ndarray:
        full = np.zeros((self.n, self.n), dtype=float)
        full[:, self.tradable_idx] = np.asarray(condensed, dtype=float)
        return full

    def expand_vector(self, condensed: np.ndarray) -> np.ndarray:
        full = np.zeros((self.n,), dtype=float)
        full[self.tradable_idx] = np.asarray(condensed, dtype=float)
        return full

    def tradable_position(self, sector: int) -> int:
        return self._tradable_pos[int(sector)]


def _weighted_relative_error(actual: np.ndarray | float, expected: np.ndarray | float, weight: float = 1.0) -> np.ndarray:
    return relative_error(actual, expected) * float(weight)


def _country_block(
    state_c: Dict[str, np.ndarray],
    state_o: Dict[str, np.ndarray],
    params_c: CountryParams,
    layout: EquilibriumLayout,
) -> Tuple[list[float], np.ndarray, float]:
    """构造单国静态残差。"""

    prices = np.asarray(state_c["price"], dtype=float)
    import_prices = np.asarray(params_c.import_cost, dtype=float) * np.asarray(state_o["price"], dtype=float)

    X_dom = np.asarray(state_c["X_dom"], dtype=float)
    X_imp = layout.expand_matrix(state_c["X_imp"])
    C_dom = np.asarray(state_c["C_dom"], dtype=float)
    C_imp = layout.expand_vector(state_c["C_imp"])

    outputs = compute_output(params_c, X_dom, X_imp, layout.tradable_mask)
    income = compute_income(params_c, prices, outputs)

    res: list[float] = []

    lambdas = compute_marginal_cost(params_c, prices, import_prices, layout.tradable_mask)
    for i in range(layout.n):
        res.append(float(relative_error(prices[i], lambdas[i])))

    # 生产端需求方程
    for i in range(layout.n):
        Pi = prices[i]
        Yi = outputs[i]
        for j in range(layout.n):
            a = float(params_c.alpha[i, j])
            if a <= 0.0:
                res.append(float(relative_error(X_dom[i, j], EPS)))
                if layout.tradable_mask[j]:
                    res.append(float(relative_error(X_imp[i, j], EPS)))
                continue

            if layout.tradable_mask[j]:
                theta = float(
                    armington_share(
                        gamma=float(params_c.gamma[i, j]),
                        p_dom=float(prices[j]),
                        p_for=float(import_prices[j]),
                        rho=float(params_c.rho[i, j]),
                    )
                )
                expected_dom = a * theta * Pi * Yi / max(prices[j], EPS)
                expected_imp = a * (1.0 - theta) * Pi * Yi / max(import_prices[j], EPS)
                res.append(float(relative_error(X_dom[i, j], expected_dom)))
                res.append(float(relative_error(X_imp[i, j], expected_imp)))
            else:
                expected_dom = a * Pi * Yi / max(prices[j], EPS)
                res.append(float(relative_error(X_dom[i, j], expected_dom)))

    # 消费端需求方程
    for j in range(layout.n):
        b = float(params_c.beta[j])
        if b <= 0.0:
            res.append(float(relative_error(C_dom[j], EPS)))
            if layout.tradable_mask[j]:
                res.append(float(relative_error(C_imp[j], EPS)))
            continue

        if layout.tradable_mask[j]:
            theta_c = float(
                armington_share(
                    gamma=float(params_c.gamma_cons[j]),
                    p_dom=float(prices[j]),
                    p_for=float(import_prices[j]),
                    rho=float(params_c.rho_cons[j]),
                )
            )
            expected_dom = b * theta_c * income / max(prices[j], EPS)
            expected_imp = b * (1.0 - theta_c) * income / max(import_prices[j], EPS)
            res.append(float(relative_error(C_dom[j], expected_dom)))
            res.append(float(relative_error(C_imp[j], expected_imp)))
        else:
            expected_dom = b * income / max(prices[j], EPS)
            res.append(float(relative_error(C_dom[j], expected_dom)))

    # 对外收支约束
    partner_X_imp = layout.expand_matrix(state_o["X_imp"])
    partner_C_imp = layout.expand_vector(state_o["C_imp"])
    exports_value = float(np.dot(prices, partner_X_imp.sum(axis=0) + partner_C_imp) + np.dot(prices, params_c.exports))
    imports_value = float(np.dot(import_prices, X_imp.sum(axis=0) + C_imp))
    res.append(float(relative_error(exports_value, imports_value)))

    # 商品市场出清（约束前 n-1 个市场）
    market_weight = 10.0
    exports_qty = partner_X_imp.sum(axis=0) + partner_C_imp + np.asarray(params_c.exports)
    for j in range(layout.n - 1):
        total_demand_j = float(X_dom[:, j].sum() + C_dom[j] + exports_qty[j])
        res.append(float(_weighted_relative_error(outputs[j], total_demand_j, market_weight)))

    return res, outputs, income


def _equilibrium_residuals(log_vector: np.ndarray, layout: EquilibriumLayout, params: TwoCountryParams) -> np.ndarray:
    """组装两国残差并加名义锚。"""
    pos = np.exp(np.asarray(log_vector, dtype=float))
    state = layout.unpack(pos)

    res_h, _, _ = _country_block(state["H"], state["F"], params.home, layout)
    res_f, _, _ = _country_block(state["F"], state["H"], params.foreign, layout)

    residuals = list(res_h) + list(res_f)
    residuals.append(float(relative_error(state["H"]["price"][0], 1.0)))
    residuals.append(float(relative_error(state["F"]["price"][0], 1.0)))
    return np.asarray(residuals, dtype=float)


def _initial_guess(params: TwoCountryParams, layout: EquilibriumLayout) -> StateMap:
    """构造静态求解初值。"""
    st = layout.new_state(fill=EPS)

    for code, c_params in (("H", params.home), ("F", params.foreign)):
        c = st[code]
        prices = c["price"]
        prices[:] = 1.0

        import_prices = np.asarray(c_params.import_cost, dtype=float) * np.ones(layout.n, dtype=float)
        base_output = np.maximum(np.asarray(c_params.A, dtype=float), 1.0)

        for i in range(layout.n):
            Pi = prices[i]
            Yi = base_output[i]
            for j in range(layout.n):
                a = float(c_params.alpha[i, j])
                if a <= 0.0:
                    continue
                if layout.tradable_mask[j]:
                    theta = float(np.clip(c_params.gamma[i, j], 1e-3, 1.0 - 1e-3))
                    idx = layout.tradable_position(j)
                    c["X_dom"][i, j] = max(a * theta * Pi * Yi / max(prices[j], EPS), EPS)
                    c["X_imp"][i, idx] = max(a * (1.0 - theta) * Pi * Yi / max(import_prices[j], EPS), EPS)
                else:
                    c["X_dom"][i, j] = max(a * Pi * Yi / max(prices[j], EPS), EPS)

        income = float(np.sum(prices * base_output * np.clip(1.0 - c_params.alpha.sum(axis=1), 1e-8, None)))
        for j in range(layout.n):
            b = float(c_params.beta[j])
            if b <= 0.0:
                continue
            if layout.tradable_mask[j]:
                theta_c = float(np.clip(c_params.gamma_cons[j], 1e-3, 1.0 - 1e-3))
                idx = layout.tradable_position(j)
                c["C_dom"][j] = max(b * theta_c * income / max(prices[j], EPS), EPS)
                c["C_imp"][idx] = max(b * (1.0 - theta_c) * income / max(import_prices[j], EPS), EPS)
            else:
                c["C_dom"][j] = max(b * income / max(prices[j], EPS), EPS)

    return st


def _build_country_block(
    state_c: Dict[str, np.ndarray],
    state_o: Dict[str, np.ndarray],
    params_c: CountryParams,
    layout: EquilibriumLayout,
) -> CountryBlock:
    prices = np.asarray(state_c["price"], dtype=float)
    x_dom = np.asarray(state_c["X_dom"], dtype=float)
    x_imp = layout.expand_matrix(state_c["X_imp"])
    c_dom = np.asarray(state_c["C_dom"], dtype=float)
    c_imp = layout.expand_vector(state_c["C_imp"])
    imp_price = np.asarray(params_c.import_cost, dtype=float) * np.asarray(state_o["price"], dtype=float)
    return CountryBlock(X_dom=x_dom, X_imp=x_imp, C_dom=c_dom, C_imp=c_imp, price=prices, imp_price=imp_price)


def _fixed_point_step_country(
    state_c: Dict[str, np.ndarray],
    state_o: Dict[str, np.ndarray],
    params_c: CountryParams,
    layout: EquilibriumLayout,
    damping: float = 0.5,
) -> Dict[str, np.ndarray]:
    """无 scipy 时的固定点单步更新。"""
    damping = float(np.clip(damping, 1e-3, 1.0))
    keep = 1.0 - damping

    prices = np.asarray(state_c["price"], dtype=float)
    import_prices = np.asarray(params_c.import_cost, dtype=float) * np.asarray(state_o["price"], dtype=float)
    x_dom_old = np.asarray(state_c["X_dom"], dtype=float)
    x_imp_old = layout.expand_matrix(state_c["X_imp"])
    c_dom_old = np.asarray(state_c["C_dom"], dtype=float)
    c_imp_old = layout.expand_vector(state_c["C_imp"])

    outputs = compute_output(params_c, x_dom_old, x_imp_old, layout.tradable_mask)
    income = compute_income(params_c, prices, outputs)
    lambdas = compute_marginal_cost(params_c, prices, import_prices, layout.tradable_mask)
    new_price = np.clip(keep * prices + damping * lambdas, EPS, None)

    # 用“静态最优需求公式”更新中间投入与消费，再做阻尼。
    x_dom_target = np.zeros_like(x_dom_old)
    x_imp_target = np.zeros_like(x_imp_old)
    for i in range(layout.n):
        Pi = float(new_price[i])
        Yi = float(outputs[i])
        for j in range(layout.n):
            a = float(params_c.alpha[i, j])
            if a <= 0.0:
                continue
            if layout.tradable_mask[j]:
                theta = float(
                    armington_share(
                        gamma=float(params_c.gamma[i, j]),
                        p_dom=float(new_price[j]),
                        p_for=float(import_prices[j]),
                        rho=float(params_c.rho[i, j]),
                    )
                )
                x_dom_target[i, j] = a * theta * Pi * Yi / max(float(new_price[j]), EPS)
                x_imp_target[i, j] = a * (1.0 - theta) * Pi * Yi / max(float(import_prices[j]), EPS)
            else:
                x_dom_target[i, j] = a * Pi * Yi / max(float(new_price[j]), EPS)

    c_dom_target = np.zeros_like(c_dom_old)
    c_imp_target = np.zeros_like(c_imp_old)
    for j in range(layout.n):
        b = float(params_c.beta[j])
        if b <= 0.0:
            continue
        if layout.tradable_mask[j]:
            theta_c = float(
                armington_share(
                    gamma=float(params_c.gamma_cons[j]),
                    p_dom=float(new_price[j]),
                    p_for=float(import_prices[j]),
                    rho=float(params_c.rho_cons[j]),
                )
            )
            c_dom_target[j] = b * theta_c * income / max(float(new_price[j]), EPS)
            c_imp_target[j] = b * (1.0 - theta_c) * income / max(float(import_prices[j]), EPS)
        else:
            c_dom_target[j] = b * income / max(float(new_price[j]), EPS)

    x_dom_new = np.clip(keep * x_dom_old + damping * x_dom_target, EPS, None)
    x_imp_new = np.clip(keep * x_imp_old + damping * x_imp_target, EPS, None)
    c_dom_new = np.clip(keep * c_dom_old + damping * c_dom_target, EPS, None)
    c_imp_new = np.clip(keep * c_imp_old + damping * c_imp_target, EPS, None)

    # 先做进口外汇约束近似校正，减少贸易差额漂移。
    partner_x_imp = layout.expand_matrix(state_o["X_imp"])
    partner_c_imp = layout.expand_vector(state_o["C_imp"])
    exports_value = float(np.dot(new_price, partner_x_imp.sum(axis=0) + partner_c_imp) + np.dot(new_price, params_c.exports))
    imports_value = float(np.dot(import_prices, x_imp_new.sum(axis=0) + c_imp_new))
    if imports_value > EPS:
        fx_scale = min(1.0, exports_value / imports_value)
        x_imp_new *= fx_scale
        c_imp_new *= fx_scale

    # 再做国内品市场出清近似校正。
    exports_qty = partner_x_imp.sum(axis=0) + partner_c_imp + np.asarray(params_c.exports)
    demand_total = x_dom_new.sum(axis=0) + c_dom_new + exports_qty
    scale_dom = np.ones(layout.n, dtype=float)
    positive = demand_total > EPS
    scale_dom[positive] = np.minimum(1.0, outputs[positive] / demand_total[positive])
    x_dom_new *= scale_dom[np.newaxis, :]
    c_dom_new *= scale_dom

    return {
        "X_dom": x_dom_new,
        "X_imp": x_imp_new[:, layout.tradable_idx],
        "C_dom": c_dom_new,
        "C_imp": c_imp_new[layout.tradable_idx],
        "price": new_price,
    }


def _solve_without_scipy(
    params: TwoCountryParams,
    layout: EquilibriumLayout,
    initial_state: StateMap,
    max_iterations: int,
    tolerance: float,
) -> tuple[StateMap, bool, int, float, str]:
    """无 scipy 环境的回退解法（固定点迭代）。"""
    state = {
        "H": {k: np.array(v, copy=True) for k, v in initial_state["H"].items()},
        "F": {k: np.array(v, copy=True) for k, v in initial_state["F"].items()},
    }
    tol = max(float(tolerance), 1e-10)
    last_res = np.inf
    converged = False

    # Phase A: 先做固定点热启动，让状态进入可行域附近。
    warmup_iters = max(10, min(int(max_iterations), 80))
    for it in range(1, warmup_iters + 1):
        old_h = {k: np.array(v, copy=True) for k, v in state["H"].items()}
        old_f = {k: np.array(v, copy=True) for k, v in state["F"].items()}

        state["H"] = _fixed_point_step_country(old_h, old_f, params.home, layout, damping=0.45)
        state["F"] = _fixed_point_step_country(old_f, state["H"], params.foreign, layout, damping=0.45)

        packed = layout.pack(state)
        res = _equilibrium_residuals(np.log(np.clip(packed, EPS, None)), layout, params)
        last_res = float(np.linalg.norm(res))
        if last_res < max(1e-2, tol * 1e3):
            converged = True
            return state, converged, it, last_res, "fixed_point_converged"

    # Phase B: 变量规模较小时，使用有限差分 LM 做精化。
    n_vars = int(layout.total_size)
    # 变量维度较大时，有限差分雅可比会非常慢且不稳定，直接返回固定点结果。
    if n_vars > 120:
        return state, converged, warmup_iters, float(last_res), "fixed_point_max_iter_reached_large_scale"

    x = np.log(np.clip(layout.pack(state), EPS, None))
    lam = 1e-2
    target = max(5e-3, tol * 1e3)
    remain_iters = max(5, int(max_iterations) - warmup_iters)
    total_it = warmup_iters

    for _ in range(remain_iters):
        r = _equilibrium_residuals(x, layout, params)
        r_norm = float(np.linalg.norm(r))
        total_it += 1
        if r_norm < target:
            converged = True
            break

        m = r.size
        n = x.size
        J = np.empty((m, n), dtype=float)
        h = 1e-5
        for k in range(n):
            xk = np.array(x, copy=True)
            xk[k] += h
            rk = _equilibrium_residuals(xk, layout, params)
            J[:, k] = (rk - r) / h

        A = J.T @ J + lam * np.eye(n, dtype=float)
        g = J.T @ r
        try:
            dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(A, -g, rcond=None)[0]

        accepted = False
        for step in (1.0, 0.5, 0.25, 0.1, 0.05):
            cand = x + step * dx
            cand_r = _equilibrium_residuals(cand, layout, params)
            cand_norm = float(np.linalg.norm(cand_r))
            if cand_norm < r_norm:
                x = cand
                last_res = cand_norm
                lam = max(lam * 0.7, 1e-8)
                accepted = True
                break

        if not accepted:
            # LM 方向未改进时，增大阻尼并尝试小步梯度下降。
            lam = min(lam * 5.0, 1e8)
            gd_step = x - 1e-3 * g
            gd_norm = float(np.linalg.norm(_equilibrium_residuals(gd_step, layout, params)))
            if gd_norm < r_norm:
                x = gd_step
                last_res = gd_norm
                accepted = True

        if not accepted and lam >= 1e7:
            break

    final_state = layout.unpack(np.exp(x))
    final_res = float(np.linalg.norm(_equilibrium_residuals(x, layout, params)))
    return final_state, bool(converged), total_it, final_res, "fixed_point_lm_refined"


def solve_static_equilibrium(
    params: TwoCountryParams | Dict[str, Dict],
    max_iterations: int = 400,
    tolerance: float = 1e-8,
) -> StaticEquilibriumResult:
    """求解两国静态均衡。

    注意：
    - 使用 log-parameterization 强制变量为正；
    - 采用 scipy least_squares 做非线性最小二乘。
    """
    if not isinstance(params, TwoCountryParams):
        params = from_raw_dict(params)

    layout = EquilibriumLayout(params.n, np.asarray(params.tradable_idx, dtype=int))
    guess = _initial_guess(params, layout)
    log_guess = np.log(np.clip(layout.pack(guess), EPS, None))

    def residual_fn(v: np.ndarray) -> np.ndarray:
        return _equilibrium_residuals(v, layout, params)

    # 优先使用 scipy；若环境缺失则自动回退固定点迭代。
    try:
        from scipy.optimize import least_squares  # type: ignore

        result = least_squares(
            residual_fn,
            log_guess,
            ftol=float(tolerance),
            xtol=float(tolerance),
            gtol=float(tolerance),
            max_nfev=int(max_iterations),
            verbose=0,
        )
        sol_vec = np.exp(result.x)
        sol_state = layout.unpack(sol_vec)
        converged = bool(result.success)
        iterations = int(result.nfev)
        final_residual = float(np.linalg.norm(residual_fn(result.x)))
        solver_message = str(result.message)
    except Exception:
        sol_state, converged, iterations, final_residual, solver_message = _solve_without_scipy(
            params=params,
            layout=layout,
            initial_state=guess,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    home_block = _build_country_block(sol_state["H"], sol_state["F"], params.home, layout)
    foreign_block = _build_country_block(sol_state["F"], sol_state["H"], params.foreign, layout)

    return StaticEquilibriumResult(
        home=home_block,
        foreign=foreign_block,
        converged=bool(converged),
        iterations=int(iterations),
        final_residual=final_residual,
        solver_message=solver_message,
    )
