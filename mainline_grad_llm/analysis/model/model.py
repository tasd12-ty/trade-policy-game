r"""核心模型与静态均衡（NumPy 实现）。

目的：保持最小可用的两国多部门静态一般均衡骨架，并清晰标注经济学含义与公式对应关系（中文注释）。

包含：
- 数据与参数：CountryParams, ModelParams, create_symmetric_parameters, normalize_model_params
- Armington/CES 工具函数：safe_log, armington_share, armington_price, armington_quantity
- 静态均衡：EquilibriumLayout, _country_block, _equilibrium_residuals, solve_initial_equilibrium

经济学要点（与 production_network_simulation0916.tex 对应）：
- 生产函数：部门 i 的产出
    Y_i = A_i \prod_j \Big[ X_{ij} \Big]^{\alpha_{ij}}，其中当 j 为可贸易部门时，使用嵌套 CES 组合
    X_{ij}^{CES} = \left[ (\gamma_{ij} X_{ij}^I)^{\rho_{ij}} + ((1-\gamma_{ij}) X_{ij}^O)^{\rho_{ij}} \right]^{\alpha_{ij}/\rho_{ij}}。
- 成本（对偶）与零利润条件：边际成本（单位成本）λ_i 满足
    ln λ_i = - ln A_i + \sum_j \alpha_{ij} ln P_j^*，其中 P_j^* 对可贸易品使用 Armington 对偶价格（见 armington_price）。
- 消费者效用：对不可贸易品为 Cobb–Douglas，对可贸易品为嵌套 CES；由此得到消费需求与预算约束。
- 市场清算与国际收支：
    · 国内品供给 Y_j 用于中间品与终端消费及出口；
    · 进口支付以出口收入（含基础出口向量）约束（见 _country_block 中贸易平衡残差）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares

EPS = 1e-9
NP_DTYPE = np.float64
DEFAULT_DEVICE = None


# ---------------------
# 数据类与参数工具
# ---------------------


@dataclass(frozen=True)
class CountryParams:
    """国家参数块（定标后均为 numpy 数组）。"""

    alpha: np.ndarray
    gamma: np.ndarray
    rho: np.ndarray
    beta: np.ndarray
    A: np.ndarray
    exports: np.ndarray
    gamma_cons: np.ndarray
    rho_cons: np.ndarray
    import_cost: np.ndarray

    def to(self, _device=None) -> "CountryParams":
        """NumPy 版本无设备概念，保持接口兼容。"""
        return self

    @property
    def device(self):
        return None


@dataclass(frozen=True)
class ModelParams:
    """模型顶层参数。"""

    home: CountryParams
    foreign: CountryParams
    tradable_idx: np.ndarray
    non_tradable_idx: np.ndarray

    def to(self, _device=None) -> "ModelParams":
        return self

    @property
    def device(self):
        return None


def create_symmetric_parameters() -> Dict[str, dict]:
    """构造对称的两国 6 部门基线参数（简化校验用）。"""
    n = 6
    alpha_base = 0.15
    alpha_H = np.full((n, n), alpha_base)
    alpha_F = np.full((n, n), alpha_base)
    np.fill_diagonal(alpha_H, 0.0)
    np.fill_diagonal(alpha_F, 0.0)

    gamma_H = np.full((n, n), 0.5)
    gamma_F = np.full((n, n), 0.5)
    tradable = [2, 3, 4, 5]
    for i in range(n):
        for j in range(n):
            if j not in tradable or i == j:
                gamma_H[i, j] = 1.0
                gamma_F[i, j] = 1.0

    rho_H = np.full((n, n), 0.2)
    rho_F = np.full((n, n), 0.2)
    beta_H = np.ones(n) / n
    beta_F = np.ones(n) / n
    A_H = np.ones(n)
    A_F = np.ones(n)
    Export_H = np.zeros(n)
    Export_F = np.zeros(n)
    gamma_c_H = gamma_H[0].copy()
    gamma_c_F = gamma_F[0].copy()
    rho_c_H = rho_H[0].copy()
    rho_c_F = rho_F[0].copy()
    import_cost = np.ones(n)

    return {
        "H": {
            "alpha_ij": alpha_H,
            "gamma_ij": gamma_H,
            "rho_ij": rho_H,
            "beta_j": beta_H,
            "A_i": A_H,
            "Export_i": Export_H,
            "gamma_cj": gamma_c_H,
            "rho_cj": rho_c_H,
            "import_cost": import_cost,
        },
        "F": {
            "alpha_ij": alpha_F,
            "gamma_ij": gamma_F,
            "rho_ij": rho_F,
            "beta_j": beta_F,
            "A_i": A_F,
            "Export_i": Export_F,
            "gamma_cj": gamma_c_F,
            "rho_cj": rho_c_F,
            "import_cost": import_cost,
        },
        "tradable_sectors": tradable,
    }


def _to_country_params(block: Dict[str, np.ndarray]) -> CountryParams:
    alpha_np = np.asarray(block["alpha_ij"], dtype=NP_DTYPE)
    gamma_np = np.asarray(block["gamma_ij"], dtype=NP_DTYPE)
    rho_np = np.asarray(block["rho_ij"], dtype=NP_DTYPE)
    beta_np = np.asarray(block["beta_j"], dtype=NP_DTYPE)
    A_np = np.asarray(block["A_i"], dtype=NP_DTYPE)
    exports_np = np.asarray(block.get("Export_i", np.zeros_like(beta_np)), dtype=NP_DTYPE)
    gamma_cons_np = np.asarray(block.get("gamma_cj", gamma_np[0]), dtype=NP_DTYPE)
    rho_cons_np = np.asarray(block.get("rho_cj", rho_np[0]), dtype=NP_DTYPE)
    import_cost_np = np.asarray(block.get("import_cost", np.ones_like(beta_np)), dtype=NP_DTYPE)

    return CountryParams(
        alpha=alpha_np,
        gamma=gamma_np,
        rho=rho_np,
        beta=beta_np,
        A=A_np,
        exports=exports_np,
        gamma_cons=gamma_cons_np,
        rho_cons=rho_cons_np,
        import_cost=import_cost_np,
    )


def normalize_model_params(raw_params) -> ModelParams:
    if isinstance(raw_params, ModelParams):
        return raw_params
    home = _to_country_params(raw_params["H"])
    foreign = _to_country_params(raw_params["F"])
    tradable = np.array(sorted(raw_params["tradable_sectors"]), dtype=int)
    n = home.alpha.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[tradable] = True
    non_tradable = np.array([i for i in range(n) if not mask[i]], dtype=int)
    return ModelParams(home=home, foreign=foreign, tradable_idx=tradable, non_tradable_idx=non_tradable)


# ---------------------
# Armington / CES 与通用数学
# ---------------------


def safe_log(x: np.ndarray) -> np.ndarray:
    """数值安全的对数：log(max(x, EPS))，避免 0 或负数导致的 NaN。"""
    return np.log(np.clip(x, EPS, None))


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=NP_DTYPE)


def armington_share(gamma, p_dom, p_for, rho):
    """Armington 份额 θ(p)：在给定价格下的“本国产品份额”。"""
    g = _as_array(gamma)
    p_d = np.clip(_as_array(p_dom), EPS, None)
    p_f = np.clip(_as_array(p_for), EPS, None)
    r = _as_array(rho)

    sigma = np.where(np.abs(1.0 - r) < 1e-10, 1.0, 1.0 / (1.0 - r))

    # σ->∞ 时取最低价；σ->1 时为 Cobb–Douglas
    mask_inf = np.abs(1.0 - r) < 1e-6
    mask_sigma1 = np.abs(sigma - 1.0) < 1e-8
    sigma_safe = np.where(mask_sigma1, 0.5, sigma)

    w_d = (g ** sigma_safe) * (p_d ** (1.0 - sigma_safe))
    w_f = ((1.0 - g) ** sigma_safe) * (p_f ** (1.0 - sigma_safe))
    denom = np.clip(w_d + w_f, EPS, None)
    share = w_d / denom
    share = np.where(mask_sigma1, g, share)
    share = np.where(mask_inf, np.where(p_d <= p_f, 1.0, 0.0), share)
    return np.clip(share, 1e-6, 1.0 - 1e-6)


def armington_price(gamma, p_dom, p_for, rho):
    """Armington 对偶价格 P^*(p)：嵌套 CES 的单位成本函数。"""
    g = _as_array(gamma)
    p_d = np.clip(_as_array(p_dom), EPS, None)
    p_f = np.clip(_as_array(p_for), EPS, None)
    r = _as_array(rho)

    sigma = np.where(np.abs(1.0 - r) < 1e-10, 1.0, 1.0 / (1.0 - r))
    mask_inf = np.abs(1.0 - r) < 1e-6
    mask_sigma1 = np.abs(sigma - 1.0) < 1e-8
    sigma_safe = np.where(mask_sigma1, 0.5, sigma)

    inner = (g ** sigma_safe) * (p_d ** (1.0 - sigma_safe)) + ((1.0 - g) ** sigma_safe) * (p_f ** (1.0 - sigma_safe))
    price = np.clip(inner, EPS, None) ** (1.0 / (1.0 - sigma_safe))
    geom = np.exp(g * np.log(p_d) + (1.0 - g) * np.log(p_f))
    price = np.where(mask_sigma1, geom, price)
    price = np.where(mask_inf, np.minimum(p_d, p_f), price)
    return price


def armington_quantity(gamma, x_dom, x_for, alpha, rho):
    """Armington 物量合成：作为生产函数的""有效投入""。"""
    if alpha <= 0.0:
        return 1.0
    g = _as_array(gamma)
    x_d = np.clip(_as_array(x_dom), EPS, None)
    x_f = np.clip(_as_array(x_for), EPS, None)
    r = float(_as_array(rho))
    if abs(r) < 1e-10:
        return np.exp(alpha * (g * np.log(x_d) + (1.0 - g) * np.log(x_f)))
    comp = g * (x_d ** r) + (1.0 - g) * (x_f ** r)
    return np.clip(comp, EPS, None) ** (alpha / r)


# ---------------------
# 生产与成本
# ---------------------


def compute_output(params: CountryParams, X_dom: np.ndarray, X_imp: np.ndarray, tradable_mask: np.ndarray) -> np.ndarray:
    """生产函数：逐部门计算 Y_i。"""
    n = params.alpha.shape[0]
    Y = np.zeros(n, dtype=NP_DTYPE)
    for i in range(n):
        prod = max(params.A[i], EPS)
        for j in range(n):
            a = float(params.alpha[i, j])
            if a <= 0.0:
                continue
            if tradable_mask[j]:
                qty = armington_quantity(params.gamma[i, j], X_dom[i, j], X_imp[i, j], a, params.rho[i, j])
                prod = prod * max(float(qty), EPS)
            else:
                comp = max(X_dom[i, j], EPS)
                prod = prod * (comp ** a)
        Y[i] = max(prod, EPS)
    return Y


def compute_marginal_cost(params: CountryParams, prices: np.ndarray, import_prices: np.ndarray,
                          tradable_mask: np.ndarray) -> np.ndarray:
    """单位成本（边际成本）λ_i：零利润条件下 P_i = λ_i。"""
    n = params.alpha.shape[0]
    lambdas = np.zeros(n, dtype=NP_DTYPE)
    for i in range(n):
        log_cost = -safe_log(params.A[i])
        for j in range(n):
            a = float(params.alpha[i, j])
            if a <= 0.0:
                continue
            if tradable_mask[j]:
                p_idx = armington_price(params.gamma[i, j], prices[j], import_prices[j], params.rho[i, j])
                log_cost = log_cost + a * safe_log(p_idx)
            else:
                log_cost = log_cost + a * safe_log(prices[j])
        lambdas[i] = float(np.exp(log_cost))
    return lambdas


def value_added_share(params: CountryParams) -> np.ndarray:
    row_sum = np.sum(params.alpha, axis=1)
    return np.clip(1.0 - row_sum, 1e-6, None)


def compute_income(params: CountryParams, prices: np.ndarray, outputs: np.ndarray) -> float:
    va = value_added_share(params)
    return float(np.sum(prices * outputs * va))


# ---------------------
# 静态均衡求解
# ---------------------


StateDict = Dict[str, Dict[str, np.ndarray]]


class EquilibriumLayout:
    def __init__(self, n: int, tradable_idx: np.ndarray):
        self.n = int(n)
        self.tradable_idx = np.array(tradable_idx, dtype=int)
        self.tradable_mask = np.zeros(self.n, dtype=bool)
        self.tradable_mask[self.tradable_idx] = True
        self.non_tradable_idx = np.array([i for i in range(self.n) if not self.tradable_mask[i]], dtype=int)
        self.n_tradable = int(len(self.tradable_idx))
        self._tradable_pos = {int(s): i for i, s in enumerate(self.tradable_idx.tolist())}

        self.block_size = self.n * self.n + self.n * self.n_tradable + self.n + self.n_tradable + self.n
        self.total_size = self.block_size * 2

    def new_state(self, fill: float = EPS) -> StateDict:
        def _block() -> Dict[str, np.ndarray]:
            return {
                "X_dom": np.full((self.n, self.n), fill, dtype=NP_DTYPE),
                "X_imp": np.full((self.n, self.n_tradable), fill, dtype=NP_DTYPE),
                "C_dom": np.full((self.n,), fill, dtype=NP_DTYPE),
                "C_imp": np.full((self.n_tradable,), fill, dtype=NP_DTYPE),
                "price": np.ones((self.n,), dtype=NP_DTYPE),
            }
        return {"H": _block(), "F": _block()}

    def pack(self, state: StateDict) -> np.ndarray:
        vec = np.empty(self.total_size, dtype=NP_DTYPE)
        offset = 0
        for label in ("H", "F"):
            c = state[label]
            sizes = [self.n * self.n, self.n * self.n_tradable, self.n, self.n_tradable, self.n]
            arrays = [c["X_dom"].ravel(), c["X_imp"].ravel(), c["C_dom"], c["C_imp"], c["price"]]
            for arr, sz in zip(arrays, sizes):
                vec[offset:offset + sz] = arr
                offset += sz
        return vec

    def unpack(self, vector: np.ndarray) -> StateDict:
        v = np.asarray(vector, dtype=NP_DTYPE)
        if v.size != self.total_size:
            raise ValueError("向量长度不匹配")
        out = {}
        offset = 0
        for label in ("H", "F"):
            X_dom = v[offset:offset + self.n * self.n].reshape(self.n, self.n)
            offset += self.n * self.n
            X_imp = v[offset:offset + self.n * self.n_tradable].reshape(self.n, self.n_tradable)
            offset += self.n * self.n_tradable
            C_dom = v[offset:offset + self.n]
            offset += self.n
            C_imp = v[offset:offset + self.n_tradable]
            offset += self.n_tradable
            price = v[offset:offset + self.n]
            offset += self.n
            out[label] = {"X_dom": X_dom, "X_imp": X_imp, "C_dom": C_dom, "C_imp": C_imp, "price": price}
        return out

    def expand_matrix(self, condensed: np.ndarray) -> np.ndarray:
        full = np.zeros((self.n, self.n), dtype=NP_DTYPE)
        full[:, self.tradable_idx] = condensed
        return full

    def expand_vector(self, condensed: np.ndarray) -> np.ndarray:
        full = np.zeros((self.n,), dtype=NP_DTYPE)
        full[self.tradable_idx] = condensed
        return full

    def tradable_position(self, sector: int) -> int:
        return self._tradable_pos[int(sector)]

    def is_tradable(self, sector: int) -> bool:
        return bool(self.tradable_mask[int(sector)])


def _relative_error(actual: float, expected: float) -> float:
    scale = max(max(abs(expected), abs(actual)), 1.0)
    return (actual - expected) / scale


def _weighted_relative_error(actual: float, expected: float, weight: float = 1.0) -> float:
    return _relative_error(actual, expected) * weight


def _country_block(state_c: Dict[str, np.ndarray], state_o: Dict[str, np.ndarray],
                   params_c: CountryParams, layout: EquilibriumLayout) -> Tuple[list, np.ndarray, float]:
    prices = state_c["price"]
    import_prices = params_c.import_cost * state_o["price"]

    X_dom = state_c["X_dom"]
    X_imp = layout.expand_matrix(state_c["X_imp"])
    C_dom = state_c["C_dom"]
    C_imp = layout.expand_vector(state_c["C_imp"])

    outputs = compute_output(params_c, X_dom, X_imp, layout.tradable_mask)
    income = compute_income(params_c, prices, outputs)

    res: list = []

    lambdas = compute_marginal_cost(params_c, prices, import_prices, layout.tradable_mask)
    for i in range(layout.n):
        res.append(_relative_error(prices[i], lambdas[i]))

    for i in range(layout.n):
        Pi, Yi = prices[i], outputs[i]
        for j in range(layout.n):
            a = float(params_c.alpha[i, j])
            if a <= 0.0:
                res.append(_relative_error(X_dom[i, j], EPS))
                if layout.is_tradable(j):
                    res.append(_relative_error(X_imp[i, j], EPS))
                continue
            if layout.is_tradable(j):
                theta = armington_share(params_c.gamma[i, j], prices[j], import_prices[j], params_c.rho[i, j])
                exp_dom = a * theta * Pi * Yi / max(prices[j], EPS)
                exp_imp = a * (1.0 - theta) * Pi * Yi / max(import_prices[j], EPS)
                res.append(_relative_error(X_dom[i, j], exp_dom))
                res.append(_relative_error(X_imp[i, j], exp_imp))
            else:
                exp_dom = a * Pi * Yi / max(prices[j], EPS)
                res.append(_relative_error(X_dom[i, j], exp_dom))

    for j in range(layout.n):
        b = float(params_c.beta[j])
        if b <= 0.0:
            res.append(_relative_error(C_dom[j], EPS))
            if layout.is_tradable(j):
                res.append(_relative_error(C_imp[j], EPS))
            continue
        if layout.is_tradable(j):
            theta_c = armington_share(params_c.gamma_cons[j], prices[j], import_prices[j], params_c.rho_cons[j])
            exp_dom = b * theta_c * income / max(prices[j], EPS)
            exp_imp = b * (1.0 - theta_c) * income / max(import_prices[j], EPS)
            res.append(_relative_error(C_dom[j], exp_dom))
            res.append(_relative_error(C_imp[j], exp_imp))
        else:
            exp_dom = b * income / max(prices[j], EPS)
            res.append(_relative_error(C_dom[j], exp_dom))

    partner_X_imp = layout.expand_matrix(state_o["X_imp"])
    partner_C_imp = layout.expand_vector(state_o["C_imp"])
    exports_value = float(np.dot(prices, partner_X_imp.sum(axis=0) + partner_C_imp) + np.dot(prices, params_c.exports))
    imports_value = float(np.dot(import_prices, X_imp.sum(axis=0) + C_imp))
    res.append(_relative_error(exports_value, imports_value))

    MARKET_CLEARING_WEIGHT = 10.0
    exports = partner_X_imp.sum(axis=0) + partner_C_imp + params_c.exports
    for j in range(layout.n - 1):
        total_demand_j = X_dom[:, j].sum() + C_dom[j] + exports[j]
        res.append(_weighted_relative_error(outputs[j], total_demand_j, MARKET_CLEARING_WEIGHT))

    return res, outputs, income


def _equilibrium_residuals(log_vec: np.ndarray, layout: EquilibriumLayout, params: ModelParams) -> np.ndarray:
    pos = np.exp(log_vec)
    state = layout.unpack(pos)
    res_H, _, _ = _country_block(state["H"], state["F"], params.home, layout)
    res_F, _, _ = _country_block(state["F"], state["H"], params.foreign, layout)
    residuals = res_H + res_F
    residuals.append(_relative_error(state["H"]["price"][0], 1.0))
    residuals.append(_relative_error(state["F"]["price"][0], 1.0))
    return np.array(residuals, dtype=NP_DTYPE)


def _initial_guess(params: ModelParams, layout: EquilibriumLayout) -> StateDict:
    """启发式初值。"""
    st = layout.new_state(fill=EPS)
    for label, c_params in (("H", params.home), ("F", params.foreign)):
        c = st[label]
        prices = c["price"]
        prices[:] = 1.0
        import_prices = c_params.import_cost * np.ones(layout.n, dtype=NP_DTYPE)
        base_output = np.maximum(c_params.A, 1.0)
        for i in range(layout.n):
            Pi, Yi = prices[i], base_output[i]
            for j in range(layout.n):
                a = float(c_params.alpha[i, j])
                if a <= 0.0:
                    continue
                if layout.is_tradable(j):
                    theta = np.clip(c_params.gamma[i, j], 1e-3, 1.0 - 1e-3)
                    idx = layout.tradable_position(j)
                    c["X_dom"][i, j] = max(a * theta * Pi * Yi / max(prices[j], EPS), EPS)
                    c["X_imp"][i, idx] = max(a * (1.0 - theta) * Pi * Yi / max(import_prices[j], EPS), EPS)
                else:
                    c["X_dom"][i, j] = max(a * Pi * Yi / max(prices[j], EPS), EPS)

        income = np.sum(prices * base_output * value_added_share(c_params))
        for j in range(layout.n):
            b = float(c_params.beta[j])
            if b <= 0.0:
                continue
            if layout.is_tradable(j):
                theta_c = np.clip(c_params.gamma_cons[j], 1e-3, 1.0 - 1e-3)
                idx = layout.tradable_position(j)
                c["C_dom"][j] = max(b * theta_c * income / max(prices[j], EPS), EPS)
                c["C_imp"][idx] = max(b * (1.0 - theta_c) * income / max(import_prices[j], EPS), EPS)
            else:
                c["C_dom"][j] = max(b * income / max(prices[j], EPS), EPS)
    return st


def _build_country_solution(state_c: Dict[str, np.ndarray], state_o: Dict[str, np.ndarray],
                            params_c: CountryParams, layout: EquilibriumLayout) -> Dict[str, Dict[str, np.ndarray]]:
    prices = state_c["price"]
    X_dom = state_c["X_dom"]
    X_imp = layout.expand_matrix(state_c["X_imp"])
    C_dom = state_c["C_dom"]
    C_imp = layout.expand_vector(state_c["C_imp"])

    X_I = np.zeros_like(X_dom)
    X_I[:, layout.tradable_idx] = X_dom[:, layout.tradable_idx]
    C_I = np.zeros_like(prices)
    C_I[layout.tradable_idx] = C_dom[layout.tradable_idx]
    C_total = np.zeros_like(prices)
    C_total[layout.non_tradable_idx] = C_dom[layout.non_tradable_idx]
    import_prices = params_c.import_cost * state_o["price"]

    return {
        "intermediate_inputs": {"X_ij": X_dom.copy(), "X_I_ij": X_I.copy(), "X_O_ij": X_imp.copy()},
        "final_consumption": {"C_j": C_total.copy(), "C_I_j": C_I.copy(), "C_O_j": C_imp.copy()},
        "prices": {"P_j": prices.copy(), "P_I_j": prices.copy(), "P_O_j": import_prices.copy()},
    }


def solve_initial_equilibrium(exogenous_params, max_iterations: int = 400, tolerance: float = 1e-8):
    """求解两国静态初始均衡（非线性最小二乘，log-变量）。"""
    params = normalize_model_params(exogenous_params)
    layout = EquilibriumLayout(params.home.alpha.shape[0], params.tradable_idx)
    guess_state = _initial_guess(params, layout)
    log_guess = np.log(np.clip(layout.pack(guess_state), EPS, None))

    def residuals(v: np.ndarray) -> np.ndarray:
        return _equilibrium_residuals(v, layout, params)

    result = least_squares(
        residuals,
        log_guess,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_iterations,
        verbose=0,
    )

    sol_vec = np.exp(result.x)
    sol_state = layout.unpack(sol_vec)
    final_resid = float(np.linalg.norm(residuals(result.x)))

    home_block = _build_country_solution(sol_state["H"], sol_state["F"], params.home, layout)
    foreign_block = _build_country_solution(sol_state["F"], sol_state["H"], params.foreign, layout)
    return {
        "H": home_block,
        "F": foreign_block,
        "convergence_info": {
            "converged": bool(result.success),
            "iterations": int(result.nfev),
            "final_residual": final_resid,
            "solver_message": str(result.message),
        },
    }


def test_simple_case():
    params = create_symmetric_parameters()
    layout_params = normalize_model_params(params)
    layout = EquilibriumLayout(layout_params.home.alpha.shape[0], layout_params.tradable_idx)
    guess = _initial_guess(layout_params, layout)
    log_vec = np.log(np.clip(layout.pack(guess), EPS, None))
    resid = _equilibrium_residuals(log_vec, layout, layout_params)
    print(f"初始残差范数: {np.linalg.norm(resid):.3e}")
    info = solve_initial_equilibrium(params, max_iterations=200, tolerance=1e-8)["convergence_info"]
    print(f"收敛: {info['converged']}, 残差: {info['final_residual']:.3e}, 迭代: {info['iterations']}")
    return info


__all__ = [
    "CountryParams",
    "ModelParams",
    "create_symmetric_parameters",
    "normalize_model_params",
    "EPS",
    "NP_DTYPE",
    "DEFAULT_DEVICE",
    "safe_log",
    "armington_share",
    "armington_price",
    "armington_quantity",
    "compute_output",
    "compute_marginal_cost",
    "compute_income",
    "solve_initial_equilibrium",
    "test_simple_case",
]
