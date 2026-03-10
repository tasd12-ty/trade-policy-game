"""动态矩阵引擎。

公式对应：
- eq 19: 价格更新 Δp = Diag(τ) · (总需求 − 产出)
- eq 20: 国内品分配（按需求比例配给）
- eq 21: 国内消费分配
- eq 22: 出口分配
- eq 23-24: 进口分配（外汇约束）
- eq 25: 产出更新（生产函数）
- eq 26: 收入更新

依赖：production.py, factors.py, demand.py, armington.py, utils.py
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import EPS, clamp_positive
from .armington import theta_from_quantities, armington_share_from_prices
from .production import compute_output, compute_marginal_cost
from .factors import compute_income
from .demand import (
    compute_delta_x_dom,
    compute_delta_x_imp,
    compute_delta_c_dom,
    compute_delta_c_imp,
    static_intermediate_demand_dom,
    static_intermediate_demand_imp,
    static_consumption_demand_dom,
    static_consumption_demand_imp,
)
from .types import CountryParams, CountryState


class CountryDynamics:
    """单国动态引擎。

    实现 eq 19-26 的完整动态更新步。
    不持有状态——每次 step 接收状态并返回新状态。

    参数：
        params: 单国参数
        tau:    价格调整速度向量 (Nl+M,)，或标量
        normalize_gap: 是否将供需缺口按产出归一化
    """

    def __init__(
        self,
        params: CountryParams,
        tau: np.ndarray | float = 0.5,
        normalize_gap: bool = False,
        numeraire: bool = False,
        quantity_damping: float = 1.0,
        walrasian: bool = False,
    ):
        self.params = params
        self.Nl = params.Nl
        self.Ml = int(params.Ml)
        self.M = int(params.M_factors)
        self.tradable_mask = params.tradable_mask
        self.numeraire = numeraire
        self.quantity_damping = quantity_damping
        self.walrasian = walrasian

        # τ 向量化
        tau_val = np.asarray(tau, dtype=float)
        if tau_val.ndim == 0:
            self.tau = np.full(self.Nl + self.M, float(tau_val), dtype=float)
        else:
            self.tau = tau_val.copy()
        self.normalize_gap = normalize_gap

    def step(
        self,
        state: CountryState,
        imp_price: np.ndarray,
        supply_cap: np.ndarray | None = None,
    ) -> CountryState:
        """执行一期动态更新。

        walrasian=True 模式：
            - 价格更新同 eq 19
            - 数量从 Leontief 精确求解（无配给漂移）

        walrasian=False 模式（默认）：
            1. 计算产出和边际成本
            2. 计算 θ（用量端 Armington 份额）
            3. 计算需求调整量 Δx, Δc
            4. 更新价格 (eq 19)
            5. 分配国内品 (eq 20-22)
            6. 分配进口品 (eq 23-24)
            7. 更新产出 (eq 25) 和收入 (eq 26)

        参数：
            state:      当前状态
            imp_price:  (Nl,) 进口品到岸价
            supply_cap: (Nl,) 可选的进口供应上限

        返回：
            新状态 CountryState
        """
        if self.walrasian:
            return self._walrasian_step(state, imp_price)

        p = self.params
        Nl, Ml, M = self.Nl, self.Ml, self.M
        imp_p = clamp_positive(imp_price)

        # ---- 1. 产出与边际成本 ----
        outputs = compute_output(
            p.A, p.alpha, state.X_dom, state.X_imp,
            p.gamma, p.rho, Ml, M,
        )
        lambdas = compute_marginal_cost(
            p.A, p.alpha, state.price, imp_p,
            p.gamma, p.rho, Ml, M,
        )

        # ---- 2. θ 用量端份额 (eq 9 / eq 15) ----
        theta_prod = theta_from_quantities(
            p.gamma, state.X_dom[:, :Nl], state.X_imp, p.rho,
        )
        # 非贸易品 θ=1
        theta_prod[:, :Ml] = 1.0

        theta_cons = theta_from_quantities(
            p.gamma_cons,
            state.C_dom,
            state.C_imp,
            p.rho_cons,
        )
        theta_cons[:Ml] = 1.0

        # ---- 3. 需求调整量 (eq 11-14, 16-18) ----
        delta_x_dom = compute_delta_x_dom(
            p.alpha, lambdas, outputs, theta_prod,
            state.price, state.X_dom, Ml, M,
        )
        delta_x_imp = compute_delta_x_imp(
            p.alpha[:, :Nl], lambdas, outputs, theta_prod,
            imp_p, state.X_imp, Ml,
        )

        # eq 16-18 中消费需求使用当期收入 I_t
        income = max(float(state.income), EPS)

        delta_c_dom = compute_delta_c_dom(
            p.beta, theta_cons, income,
            state.price, state.C_dom, Ml,
        )
        delta_c_imp = compute_delta_c_imp(
            p.beta, theta_cons, income,
            imp_p, state.C_imp, Ml,
        )

        # 计划需求量 = 当前量 × exp(η·Δ)
        # quantity_damping η < 1 使数量缓慢趋向最优，避免供需失衡
        eta = self.quantity_damping
        planned_X_dom = state.X_dom * np.exp(eta * delta_x_dom)
        planned_X_imp = state.X_imp * np.exp(eta * delta_x_imp)
        planned_C_dom = state.C_dom * np.exp(eta * delta_c_dom)
        planned_C_imp = state.C_imp * np.exp(eta * delta_c_imp)

        # ---- 4. 价格更新 (eq 19) ----
        new_price = self._update_prices(
            state, outputs, planned_X_dom, planned_C_dom,
        )

        # ---- 5. 国内品分配 (eq 20-22) ----
        # 使用 export_base（外生计划出口量）作为配给输入，
        # 与价格更新中使用的出口需求一致
        planned_export = np.array(state.export_base, copy=True)
        actual_X_dom, actual_C_dom, actual_export = self._allocate_domestic(
            outputs, planned_X_dom, planned_C_dom, planned_export,
        )

        # ---- 6. 进口品分配 (eq 23-24) ----
        export_value = float(np.dot(actual_export[:Nl], state.price[:Nl]))
        actual_X_imp, actual_C_imp = self._allocate_imports(
            imp_p, planned_X_imp, planned_C_imp,
            export_value, supply_cap,
        )

        # ---- 7. 产出更新 (eq 25) ----
        new_output_prod = compute_output(
            p.A, p.alpha, actual_X_dom, actual_X_imp,
            p.gamma, p.rho, Ml, M,
        )
        # 完整 output 向量 (Nl+M)：产出 + 要素禀赋
        new_output = np.concatenate([new_output_prod, np.asarray(p.L, dtype=float)])

        # ---- 8. 收入更新 (eq 26) ----
        # I_{t+1} = 1^T · X_{t+1;·,Nl+1:Nl+M} · exp(p_t)_{Nl+1:Nl+M}
        if M > 0:
            factor_usage = actual_X_dom[:, Nl:Nl + M].sum(axis=0)
            new_income = float(np.dot(factor_usage, state.price[Nl:Nl + M]))
        else:
            new_income = 0.0
        new_income = max(new_income, EPS)

        return CountryState(
            X_dom=actual_X_dom,
            X_imp=actual_X_imp,
            C_dom=actual_C_dom,
            C_imp=actual_C_imp,
            price=new_price,
            imp_price=imp_p,
            export_base=state.export_base.copy(),
            export_actual=actual_export,
            output=new_output,
            income=float(new_income),
        )

    # ---- eq 19: 价格更新 ----

    def _update_prices(
        self,
        state: CountryState,
        outputs: np.ndarray,
        planned_X_dom: np.ndarray,
        planned_C_dom: np.ndarray,
    ) -> np.ndarray:
        """价格更新 (eq 19)。

        Δp = Diag(τ) · ((X^T ∘ exp(Δx))^T · 1 + C · exp(Δc) + Export − Y)

        简化为：P' = P · exp(τ · gap)
        其中 gap = (总计划需求 − 产出) / [产出 if normalized]
        """
        Nl = self.Nl

        # 出口需求使用外生 Export_t（对应状态中的 export_base）
        export_demand = state.export_base[:Nl]

        # 计划总需求（国内品）
        planned_total = np.zeros(Nl + self.M, dtype=float)
        planned_total[:Nl] = (
            planned_X_dom[:, :Nl].sum(axis=0)
            + planned_C_dom
            + export_demand
        )
        # 要素需求
        if self.M > 0:
            planned_total[Nl:] = planned_X_dom[:, Nl:].sum(axis=0)

        # 供给
        supply = np.concatenate([outputs, np.asarray(self.params.L, dtype=float)])

        # 供需缺口
        gap = planned_total - supply

        if self.normalize_gap:
            gap = gap / np.maximum(supply, EPS)

        # 严格按 eq 19：Δp = Diag(τ) · gap
        delta = self.tau * gap

        new_price = clamp_positive(state.price * np.exp(delta))

        # 可选：固定第一个要素价格作为记账单位（numeraire）
        if self.numeraire and self.M > 0:
            w = new_price[Nl]
            if w > EPS:
                new_price = new_price / w

        return new_price

    # ---- eq 20-22: 国内品分配 ----

    def _allocate_domestic(
        self,
        outputs: np.ndarray,
        planned_X_dom: np.ndarray,
        planned_C_dom: np.ndarray,
        planned_export: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """国内品按比例配给 (eq 20-22)。

        当总计划需求 > 产出时，按需求比例缩减。
        s_j = min(1, Y_j / D_j^planned)
        """
        Nl = self.Nl

        # 扩展产出到 Nl+M（要素禀赋固定，不需配给）
        full_supply = np.concatenate([
            np.asarray(outputs, dtype=float),
            np.asarray(self.params.L, dtype=float),
        ])

        # 总计划需求
        total_demand = np.zeros(Nl + self.M, dtype=float)
        total_demand[:Nl] = (
            planned_X_dom[:, :Nl].sum(axis=0)
            + planned_C_dom
            + planned_export[:Nl]
        )
        if self.M > 0:
            total_demand[Nl:] = planned_X_dom[:, Nl:].sum(axis=0)

        # 配给比例
        scale = np.minimum(1.0, full_supply / np.maximum(total_demand, EPS))

        # 分配
        actual_X_dom = planned_X_dom * scale[np.newaxis, :]
        actual_C_dom = planned_C_dom * scale[:Nl]
        actual_export = planned_export.copy()
        actual_export[:Nl] *= scale[:Nl]

        return (
            np.maximum(actual_X_dom, 0.0),
            np.maximum(actual_C_dom, 0.0),
            np.maximum(actual_export, 0.0),
        )

    # ---- eq 23-24: 进口品分配 ----

    def _allocate_imports(
        self,
        imp_price: np.ndarray,
        planned_X_imp: np.ndarray,
        planned_C_imp: np.ndarray,
        export_value: float,
        supply_cap: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """进口品外汇约束分配 (eq 23-24)。

        外汇约束：进口总值 ≤ 出口收入
        s_fx = min(1, EI / V_imp^planned)

        可选的供应上限 supply_cap：限制单品种进口量。
        """
        # 计划进口总值
        planned_imp_value = float(
            np.sum(planned_X_imp * imp_price[np.newaxis, :])
            + np.sum(planned_C_imp * imp_price)
        )

        # 外汇缩放
        fx_scale = 1.0
        if planned_imp_value > EPS:
            fx_scale = min(1.0, max(float(export_value), 0.0) / planned_imp_value)

        actual_X_imp = planned_X_imp * fx_scale
        actual_C_imp = planned_C_imp * fx_scale

        # 供应上限
        if supply_cap is not None:
            cap = clamp_positive(supply_cap)
            total_imp = actual_X_imp.sum(axis=0) + actual_C_imp
            cap_scale = np.minimum(1.0, cap / np.maximum(total_imp, EPS))
            # 仅对可贸易品施加上限
            cap_scale[:self.Ml] = 1.0
            actual_X_imp *= cap_scale[np.newaxis, :]
            actual_C_imp *= cap_scale

        # 非贸易品严格不进口
        actual_X_imp[:, :self.Ml] = 0.0
        actual_C_imp[:self.Ml] = 0.0

        return np.maximum(actual_X_imp, 0.0), np.maximum(actual_C_imp, 0.0)

    # ---- Walrasian 模式：价格动态 + Leontief 数量求解 ----

    def _walrasian_step(
        self,
        state: CountryState,
        imp_price: np.ndarray,
    ) -> CountryState:
        """Walrasian 动态步：价格渐进调整，数量从 Leontief 精确求解。

        避免配给累积漂移。每步重新求解市场出清数量。

        步骤：
        1. 从当前价格计算 Armington 份额（价格端）
        2. 计算静态最优需求 → 总需求
        3. 供需缺口 → 价格更新 (eq 19)
        4. 用新价格重新计算 Leontief → 出清数量
        5. 用静态需求函数分配投入品
        """
        p = self.params
        Nl, Ml, M = self.Nl, self.Ml, self.M
        imp_p = clamp_positive(imp_price)

        alpha = np.asarray(p.alpha, dtype=float)
        gamma = np.asarray(p.gamma, dtype=float)
        rho_arr = np.asarray(p.rho, dtype=float)
        beta = np.asarray(p.beta, dtype=float)
        gc = np.asarray(p.gamma_cons, dtype=float)
        rc = np.asarray(p.rho_cons, dtype=float)
        L = np.asarray(p.L, dtype=float)

        P = state.price[:Nl].copy()
        w = state.price[Nl:].copy()

        # ---- 1. Armington 份额（价格端） ----
        theta = np.ones((Nl, Nl), dtype=float)
        theta_c = np.ones(Nl, dtype=float)
        for j in range(Ml, Nl):
            for i in range(Nl):
                theta[i, j] = float(armington_share_from_prices(
                    gamma[i, j], P[j], imp_p[j], rho_arr[i, j]))
            theta_c[j] = float(armington_share_from_prices(
                gc[j], P[j], imp_p[j], rc[j]))

        # ---- 2. 当前供给 = 生产函数产出 ----
        outputs = compute_output(
            p.A, alpha, state.X_dom, state.X_imp,
            gamma, rho_arr, Ml, M,
        )

        # ---- 3. 静态最优需求 at 当前价格 ----
        price_full = np.concatenate([P, w])
        income = compute_income(price_full, L, Nl)

        opt_X_dom = static_intermediate_demand_dom(
            alpha, theta, price_full, outputs, Ml, M)
        opt_C_dom = static_consumption_demand_dom(
            beta, theta_c, income, price_full, Ml)

        # 总需求
        total_demand = np.zeros(Nl + M, dtype=float)
        total_demand[:Nl] = (
            opt_X_dom[:, :Nl].sum(axis=0)
            + opt_C_dom
            + state.export_base[:Nl]
        )
        if M > 0:
            total_demand[Nl:] = opt_X_dom[:, Nl:].sum(axis=0)

        supply = np.concatenate([outputs, L])

        # ---- 4. 价格更新 (eq 19) ----
        gap = total_demand - supply
        if self.normalize_gap:
            gap = gap / np.maximum(supply, EPS)
        delta = self.tau * gap
        new_price = clamp_positive(state.price * np.exp(delta))

        if self.numeraire and M > 0:
            w_val = new_price[Nl]
            if w_val > EPS:
                new_price = new_price / w_val

        # ---- 5. Leontief 求解 at 新价格 ----
        new_P = new_price[:Nl]
        new_w = new_price[Nl:]
        new_imp_p = imp_p  # 进口价格由外部决定，本步不变

        # Armington at new prices
        new_theta = np.ones((Nl, Nl), dtype=float)
        new_theta_c = np.ones(Nl, dtype=float)
        for j in range(Ml, Nl):
            for i in range(Nl):
                new_theta[i, j] = float(armington_share_from_prices(
                    gamma[i, j], new_P[j], new_imp_p[j], rho_arr[i, j]))
            new_theta_c[j] = float(armington_share_from_prices(
                gc[j], new_P[j], new_imp_p[j], rc[j]))

        # 值 Leontief: B[j,i] = alpha[i,j] * theta[i,j]
        B = np.zeros((Nl, Nl), dtype=float)
        for i in range(Nl):
            for j in range(Nl):
                B[j, i] = alpha[i, j] * (new_theta[i, j] if j >= Ml else 1.0)

        PY_C = np.array([
            beta[j] * (new_theta_c[j] if j >= Ml else 1.0) * income
            for j in range(Nl)
        ])
        PY_E = new_P * state.export_base[:Nl]

        try:
            L_inv = np.linalg.inv(np.eye(Nl) - B)
        except np.linalg.LinAlgError:
            L_inv = np.eye(Nl)

        PY_eq = np.maximum(L_inv @ (PY_C + PY_E), EPS)
        new_output_eq = np.maximum(PY_eq / np.maximum(new_P, EPS), EPS)

        # ---- 6. 分配投入品（静态需求） ----
        new_X_dom = static_intermediate_demand_dom(
            alpha, new_theta, new_price, new_output_eq, Ml, M)
        new_X_imp = static_intermediate_demand_imp(
            alpha[:, :Nl], new_theta, new_imp_p, new_price, new_output_eq, Ml)
        new_C_dom = static_consumption_demand_dom(
            beta, new_theta_c, state.income, new_price, Ml)
        new_C_imp = static_consumption_demand_imp(
            beta, new_theta_c, state.income, new_imp_p, Ml)

        # eq 26: 用 t 期要素价格与 t+1 期要素使用量更新收入
        if M > 0:
            factor_usage = new_X_dom[:, Nl:Nl + M].sum(axis=0)
            new_income = float(np.dot(factor_usage, state.price[Nl:Nl + M]))
        else:
            new_income = 0.0
        new_income = max(new_income, EPS)

        # 出口实际量 = export_base（Walrasian 模式不配给出口）
        new_export = state.export_base.copy()

        new_output = np.concatenate([new_output_eq, L.copy()])

        return CountryState(
            X_dom=new_X_dom,
            X_imp=new_X_imp,
            C_dom=new_C_dom,
            C_imp=new_C_imp,
            price=new_price,
            imp_price=new_imp_p,
            export_base=state.export_base.copy(),
            export_actual=new_export,
            output=new_output,
            income=float(new_income),
        )
