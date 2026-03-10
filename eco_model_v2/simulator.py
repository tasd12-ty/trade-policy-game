"""两国动态仿真器。

将 CountryDynamics × 2 + SupplyChainNetwork 组装为完整仿真引擎。

设计：
- 不持有参数——参数可在仿真过程中被外部修改（政策变化）
- 持有历史记录，支持分叉(fork)用于前瞻模拟
- 支持供应链覆盖层（可选）

依赖：dynamics.py, supply_chain.py, types.py, equilibrium.py
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import CountryParams, TwoCountryParams, CountryState
from .dynamics import CountryDynamics
from .supply_chain import SupplyChainNetwork
from .factors import compute_income
from .production import compute_output
from .utils import EPS


class TwoCountrySimulator:
    """两国动态仿真器。

    用法：
        sim = TwoCountrySimulator(params, tau=0.3)
        sim.initialize()       # 从简单初值开始
        sim.run(100)           # 运行 100 期
        sim.apply_tariff("H", {3: 0.2})  # H 国对部门 3 加 20% 关税
        sim.run(50)            # 再跑 50 期
        history = sim.history  # 获取历史记录

    属性：
        params:        两国参数（可被修改）
        history:       {"H": [CountryState, ...], "F": [CountryState, ...]}
        supply_chain:  可选的供应链网络
    """

    def __init__(
        self,
        params: TwoCountryParams,
        tau: float | np.ndarray = 0.3,
        normalize_gap: bool = False,
        supply_chain: SupplyChainNetwork | None = None,
        numeraire: bool = False,
        quantity_damping: float = 1.0,
        walrasian: bool = False,
    ):
        self._params = params
        self._tau = tau
        self._normalize_gap = normalize_gap
        self._numeraire = numeraire
        self._quantity_damping = quantity_damping
        self._walrasian = walrasian
        self.supply_chain = supply_chain

        # 关税语义：设置“税率水平”而非增量累乘
        # import_cost = base_import_cost * (1 + tariff_level)
        self._base_import_cost: Dict[str, np.ndarray] = {
            "H": np.asarray(params.home.import_cost, dtype=float).copy(),
            "F": np.asarray(params.foreign.import_cost, dtype=float).copy(),
        }
        self._tariff_level: Dict[str, np.ndarray] = {
            "H": np.zeros(params.home.Nl, dtype=float),
            "F": np.zeros(params.foreign.Nl, dtype=float),
        }

        # 构建动态引擎
        self._rebuild_engines()

        # 历史记录
        self.history: Dict[str, List[CountryState]] = {"H": [], "F": []}

    def _rebuild_engines(self) -> None:
        """重建动态引擎（参数变化后调用）。"""
        self._engine_h = CountryDynamics(
            self._params.home, tau=self._tau,
            normalize_gap=self._normalize_gap,
            numeraire=self._numeraire,
            quantity_damping=self._quantity_damping,
            walrasian=self._walrasian,
        )
        self._engine_f = CountryDynamics(
            self._params.foreign, tau=self._tau,
            normalize_gap=self._normalize_gap,
            numeraire=self._numeraire,
            quantity_damping=self._quantity_damping,
            walrasian=self._walrasian,
        )

    @property
    def params(self) -> TwoCountryParams:
        return self._params

    @property
    def current_state(self) -> Dict[str, CountryState]:
        """当前（最新）状态。"""
        return {
            "H": self.history["H"][-1] if self.history["H"] else None,
            "F": self.history["F"][-1] if self.history["F"] else None,
        }

    @property
    def t(self) -> int:
        """当前时期数。"""
        return len(self.history["H"])

    # ---- 初始化 ----

    def initialize(
        self,
        home_state: CountryState | None = None,
        foreign_state: CountryState | None = None,
    ) -> None:
        """初始化仿真状态。

        如果未提供状态，使用简单的均匀初值。
        两国模型采用联合均衡迭代以消除初始进口价格不连续。
        """
        if home_state is not None and foreign_state is not None:
            self.history = {"H": [home_state.copy()], "F": [foreign_state.copy()]}
            return

        # Phase 1: 独立初始化（各国自体参考价格）
        h_state = self._make_initial_state(self._params.home)
        f_state = self._make_initial_state(self._params.foreign)

        # Phase 2: 联合均衡迭代——用对方价格作为进口价格
        Nl_h = self._params.home.Nl
        Nl_f = self._params.foreign.Nl
        for _joint in range(10):
            h_old = h_state.price[:Nl_h].copy()
            f_old = f_state.price[:Nl_f].copy()

            h_state = self._make_initial_state(
                self._params.home, foreign_prices=f_state.price[:Nl_f])
            f_state = self._make_initial_state(
                self._params.foreign, foreign_prices=h_state.price[:Nl_h])

            h_change = float(np.max(np.abs(
                h_state.price[:Nl_h] / np.maximum(h_old, EPS) - 1.0)))
            f_change = float(np.max(np.abs(
                f_state.price[:Nl_f] / np.maximum(f_old, EPS) - 1.0)))
            if max(h_change, f_change) < 1e-8:
                break

        self.history = {"H": [h_state.copy()], "F": [f_state.copy()]}

    def initialize_from_equilibrium(self) -> None:
        """从静态均衡初始化（如果求解成功）。

        .. deprecated::
            scipy least_squares 在真实 IO 数据上容易陷入局部极小值，
            均衡条件（P=MC、市场出清）误差可达 2-50%。
            推荐使用 initialize()，其直接对数法可达机器精度。
            详见 io-final/INIT_COMPARISON.md。
        """
        import warnings
        warnings.warn(
            "initialize_from_equilibrium() 在真实IO数据上收敛不可靠，"
            "推荐使用 initialize()。详见 io-final/INIT_COMPARISON.md",
            DeprecationWarning,
            stacklevel=2,
        )
        from .equilibrium import solve_static_equilibrium

        result = solve_static_equilibrium(self._params)
        h_state = self._block_to_state(result.home, self._params.home)
        f_state = self._block_to_state(result.foreign, self._params.foreign)
        self.history = {"H": [h_state], "F": [f_state]}

    # ---- 仿真运行 ----

    def run(self, periods: int) -> None:
        """运行 periods 期动态仿真。"""
        if not self.history["H"]:
            raise RuntimeError("请先调用 initialize() 或 initialize_from_equilibrium()")

        for _ in range(periods):
            self._step()

    def _step(self) -> None:
        """执行一期仿真。"""
        state_h = self.history["H"][-1]
        state_f = self.history["F"][-1]

        Nl_h = self._params.home.Nl
        Nl_f = self._params.foreign.Nl

        # 计算进口到岸价
        imp_h = (
            np.asarray(self._params.home.import_cost, dtype=float)
            * state_f.price[:Nl_f]
        )
        imp_f = (
            np.asarray(self._params.foreign.import_cost, dtype=float)
            * state_h.price[:Nl_h]
        )

        # 供应链覆盖层（eq 27: 同时映射价格和数量）
        # 使用临时状态副本，避免修改历史记录中的状态
        step_h = state_h
        step_f = state_f
        if self.supply_chain is not None and len(self.supply_chain.edges) > 0:
            sc_result = self.supply_chain.apply(
                state_h.price, state_f.price,
                state_h.export_actual, state_f.export_actual,
                imp_h, imp_f,
            )
            imp_h = sc_result["home_imp_price"]
            imp_f = sc_result["foreign_imp_price"]
            # 创建临时副本写入变换后出口量，不修改历史状态
            step_h = state_h.copy()
            step_f = state_f.copy()
            step_h.export_base[:len(sc_result["home_export"])] = sc_result["home_export"]
            step_f.export_base[:len(sc_result["foreign_export"])] = sc_result["foreign_export"]

        # 各国动态步
        new_h = self._engine_h.step(step_h, imp_h)
        new_f = self._engine_f.step(step_f, imp_f)

        self.history["H"].append(new_h)
        self.history["F"].append(new_f)

    # ---- 政策接口 ----

    def apply_tariff(
        self,
        country: str,
        tariff_rates: Dict[int, float],
    ) -> None:
        """对指定国家施加关税变更。

        语义为“设置关税率水平”：
        import_cost = base_import_cost × (1 + tariff_rate)。
        重复施加同一税率不会累乘。
        """
        from .presets import apply_tariff

        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 必须为 'H' 或 'F'")

        # 仅覆盖指定部门，未指定部门保持当前关税水平
        current = self._tariff_level[c].copy()
        Nl = len(current)
        for sector, rate in tariff_rates.items():
            if 0 <= sector < Nl:
                current[sector] = max(float(rate), 0.0)
        self._tariff_level[c] = current

        # 传入全量税率，按 base_import_cost 计算 import_cost
        full_rates = {j: float(current[j]) for j in range(Nl)}
        base_ic = self._base_import_cost[c]

        if c == "H":
            new_home = apply_tariff(
                self._params.home,
                full_rates,
                base_import_cost=base_ic,
            )
            self._params = TwoCountryParams(home=new_home, foreign=self._params.foreign)
        else:
            new_foreign = apply_tariff(
                self._params.foreign,
                full_rates,
                base_import_cost=base_ic,
            )
            self._params = TwoCountryParams(home=self._params.home, foreign=new_foreign)

        self._rebuild_engines()

    def apply_quota(
        self,
        country: str,
        quota_multipliers: Dict[int, float],
    ) -> None:
        """对指定国家施加出口配额变更。

        修改 exports 参数并重建引擎。
        同时更新最新状态的 export_base。
        """
        from .presets import apply_quota

        if country == "H":
            new_home = apply_quota(self._params.home, quota_multipliers)
            self._params = TwoCountryParams(home=new_home, foreign=self._params.foreign)
            # 更新最新状态的 export_base
            if self.history["H"]:
                self.history["H"][-1].export_base = np.array(
                    new_home.exports, copy=True, dtype=float
                )
        else:
            new_foreign = apply_quota(self._params.foreign, quota_multipliers)
            self._params = TwoCountryParams(home=self._params.home, foreign=new_foreign)
            if self.history["F"]:
                self.history["F"][-1].export_base = np.array(
                    new_foreign.exports, copy=True, dtype=float
                )

        self._rebuild_engines()

    # ---- 分叉 ----

    def fork(self) -> TwoCountrySimulator:
        """分叉仿真器，用于前瞻模拟。

        返回独立的仿真器副本，修改不影响原仿真器。
        """
        new_sim = TwoCountrySimulator.__new__(TwoCountrySimulator)
        new_sim._params = self._params  # frozen dataclass, safe to share
        new_sim._tau = self._tau
        new_sim._normalize_gap = self._normalize_gap
        new_sim._numeraire = self._numeraire
        new_sim._quantity_damping = self._quantity_damping
        new_sim._walrasian = self._walrasian
        new_sim._base_import_cost = {
            "H": self._base_import_cost["H"].copy(),
            "F": self._base_import_cost["F"].copy(),
        }
        new_sim._tariff_level = {
            "H": self._tariff_level["H"].copy(),
            "F": self._tariff_level["F"].copy(),
        }
        # 深拷贝供应链，防止变异泄漏
        new_sim.supply_chain = copy.deepcopy(self.supply_chain) if self.supply_chain else None
        new_sim._rebuild_engines()
        # 深拷贝最新状态（仅保留最后一个）
        new_sim.history = {
            "H": [self.history["H"][-1].copy()] if self.history["H"] else [],
            "F": [self.history["F"][-1].copy()] if self.history["F"] else [],
        }
        return new_sim

    # ---- 观测与度量 ----

    def get_observation(self, country: str) -> Dict:
        """获取当前经济观测。

        返回包含关键经济指标的字典，供策略智能体使用。
        """
        c = country.upper()
        opp = "F" if c == "H" else "H"

        state = self.history[c][-1]
        opp_state = self.history[opp][-1]
        cp = self._params.home if c == "H" else self._params.foreign
        opp_cp = self._params.foreign if c == "H" else self._params.home

        Nl = cp.Nl

        return {
            "country": c,
            "period": self.t,
            "income": state.income,
            "price": state.price[:Nl].tolist(),
            "factor_price": state.price[Nl:].tolist(),
            "output": state.output[:Nl].tolist(),
            "trade_balance": self._trade_balance(c),
            "import_cost": cp.import_cost.tolist(),
            "current_tariff": {
                int(j): float(r) for j, r in enumerate(self._tariff_level[c]) if r > 0.0
            },
            "exports": state.export_actual[:Nl].tolist(),
            "opponent_income": opp_state.income,
            "opponent_price": opp_state.price[:Nl].tolist(),
            "opponent_import_cost": opp_cp.import_cost.tolist(),
            "opponent_tariff": {
                int(j): float(r) for j, r in enumerate(self._tariff_level[opp]) if r > 0.0
            },
        }

    def compute_payoff(
        self,
        country: str,
        start_period: int = 0,
        *,
        income_weight: float = 1.0,
        trade_weight: float = 0.5,
        price_stability_weight: float = 0.2,
    ) -> float:
        """计算效用/收益。

        加权组合：
        - 收入增长率 = (income_final / income_initial) - 1
        - 贸易差额 / 初始收入
        - 价格稳定性 = -std(price_index)
        """
        c = country.upper()
        hist = self.history[c]

        if len(hist) <= start_period + 1:
            return 0.0

        initial = hist[start_period]
        final = hist[-1]

        # 收入增长
        income_growth = (final.income / max(initial.income, EPS)) - 1.0

        # 贸易差额
        tb = self._trade_balance(c)
        trade_score = tb / max(abs(initial.income), EPS)

        # 价格稳定性
        cp = self._params.home if c == "H" else self._params.foreign
        Nl = cp.Nl
        prices = np.array([s.price[:Nl] for s in hist[start_period:]])
        price_index = prices.mean(axis=1)
        price_stability = -float(np.std(price_index)) if len(price_index) > 1 else 0.0

        return float(
            income_weight * income_growth
            + trade_weight * trade_score
            + price_stability_weight * price_stability
        )

    def _trade_balance(self, country: str) -> float:
        """计算贸易差额 = 出口值 - 进口值。"""
        c = country.upper()
        state = self.history[c][-1]
        cp = self._params.home if c == "H" else self._params.foreign
        Nl = cp.Nl

        export_value = float(np.dot(
            state.price[:Nl],
            state.export_actual[:Nl],
        ))
        import_value = float(np.dot(
            state.imp_price,
            state.X_imp.sum(axis=0) + state.C_imp,
        ))
        return export_value - import_value

    # ---- 内部工具 ----

    def _make_initial_state(
        self,
        cp: CountryParams,
        foreign_prices: np.ndarray | None = None,
    ) -> CountryState:
        """从参数构造均衡一致的初始状态。

        求解步骤：
        1. 外层：要素价格迭代 w_k
        2. P = MC 价格求解（直接法或迭代法）
        3. 值 Leontief: (I−B) PY = PY_C + λ·PY_E
        4. λ 使总要素市场出清
        5. compute_output 验证一致性

        参数：
            cp:              单国参数
            foreign_prices:  (Nl,) 对方国产品价格；None 时使用 ic*P_dom
        """
        from .armington import armington_share_from_prices, armington_price, ces_price_norm
        from .production import compute_output, compute_marginal_cost
        from .demand import (
            static_intermediate_demand_dom,
            static_intermediate_demand_imp,
            static_consumption_demand_dom,
            static_consumption_demand_imp,
        )

        Nl = cp.Nl
        Ml = int(cp.Ml)
        M = cp.M_factors
        alpha = np.asarray(cp.alpha, dtype=float)
        gamma = np.asarray(cp.gamma, dtype=float)
        rho_arr = np.asarray(cp.rho, dtype=float)
        beta = np.asarray(cp.beta, dtype=float)
        A = np.asarray(cp.A, dtype=float)
        L = np.asarray(cp.L, dtype=float)
        ic = np.asarray(cp.import_cost, dtype=float)
        gc = np.asarray(cp.gamma_cons, dtype=float)
        rc = np.asarray(cp.rho_cons, dtype=float)
        E_0 = np.asarray(cp.exports, dtype=float)[:Nl]
        alpha_f = alpha[:, Nl:]                # (Nl, M)

        # 是否使用跨国价格
        use_foreign = foreign_prices is not None
        p_for = np.asarray(foreign_prices, dtype=float) if use_foreign else None

        # ---- CES 归一化常数（自体参考，用于直接求解初值） ----
        ces_c = np.zeros((Nl, Nl), dtype=float)
        for i in range(Nl):
            for j in range(Ml, Nl):
                ces_c[i, j] = float(ces_price_norm(
                    gamma[i, j], rho_arr[i, j], ic[j]))

        # ---- 外层：要素价格迭代 ----
        w_vec = np.ones(M, dtype=float)

        for _outer in range(60):
            # ---- P = MC 价格求解 ----
            if not use_foreign:
                # 直接对数法（精确，自体参考 imp_price = ic * P）
                B_log = alpha[:, :Nl].copy()
                rhs = np.zeros(Nl, dtype=float)
                for i in range(Nl):
                    rhs[i] = -float(np.log(max(A[i], EPS)))
                    for j in range(Nl + M):
                        a = alpha[i, j]
                        if a <= 0.0:
                            continue
                        rhs[i] -= a * float(np.log(max(a, EPS)))
                        if j >= Nl:
                            rhs[i] += a * float(np.log(max(w_vec[j - Nl], EPS)))
                        elif j >= Ml:
                            rhs[i] += a * ces_c[i, j]
                ln_P = np.linalg.solve(np.eye(Nl) - B_log, rhs)
                P = np.exp(ln_P)
            else:
                # 迭代法（跨国进口价格，imp_price = ic * P_foreign）
                if _outer == 0:
                    # 用直接法获取初始价格
                    B_log = alpha[:, :Nl].copy()
                    rhs = np.zeros(Nl, dtype=float)
                    for i in range(Nl):
                        rhs[i] = -float(np.log(max(A[i], EPS)))
                        for j in range(Nl + M):
                            a = alpha[i, j]
                            if a <= 0.0:
                                continue
                            rhs[i] -= a * float(np.log(max(a, EPS)))
                            if j >= Nl:
                                rhs[i] += a * float(np.log(max(w_vec[j - Nl], EPS)))
                            elif j >= Ml:
                                rhs[i] += a * ces_c[i, j]
                    ln_P = np.linalg.solve(np.eye(Nl) - B_log, rhs)
                    P = np.exp(ln_P)

                imp_iter = ic * p_for
                price_iter = np.ones(Nl + M, dtype=float)
                price_iter[:Nl] = P
                price_iter[Nl:] = w_vec
                for _mc in range(200):
                    mc = compute_marginal_cost(
                        A, alpha, price_iter, imp_iter,
                        gamma, rho_arr, Ml, M)
                    price_iter[:Nl] = 0.5 * price_iter[:Nl] + 0.5 * mc
                    price_iter[Nl:] = w_vec
                P = price_iter[:Nl].copy()

            price = np.ones(Nl + M, dtype=float)
            price[:Nl] = P
            price[Nl:] = w_vec
            imp_price = ic * p_for if use_foreign else ic * P

            # ---- Armington 份额（价格端） ----
            theta = np.ones((Nl, Nl), dtype=float)
            theta_c = np.ones(Nl, dtype=float)
            for i in range(Nl):
                for j in range(Ml, Nl):
                    theta[i, j] = float(armington_share_from_prices(
                        gamma[i, j], P[j], imp_price[j], rho_arr[i, j]))
            for j in range(Ml, Nl):
                theta_c[j] = float(armington_share_from_prices(
                    gc[j], P[j], imp_price[j], rc[j]))

            income = float(np.dot(w_vec, L))

            # ---- 值 Leontief: B[j,i] = α_{ij}·θ_{ij} ----
            B = np.zeros((Nl, Nl), dtype=float)
            for i in range(Nl):
                for j in range(Nl):
                    B[j, i] = alpha[i, j] * (theta[i, j] if j >= Ml else 1.0)

            PY_C = np.array([
                beta[j] * (theta_c[j] if j >= Ml else 1.0) * income
                for j in range(Nl)
            ])
            PY_E = P * E_0

            try:
                L_inv = np.linalg.inv(np.eye(Nl) - B)
            except np.linalg.LinAlgError:
                L_inv = np.eye(Nl)

            PY_cons = L_inv @ PY_C
            PY_exp = L_inv @ PY_E

            # λ: 总要素市场出清
            fac_supply = float(np.dot(w_vec, L))
            a_total = float((alpha_f * PY_cons[:, None]).sum())
            b_total = float((alpha_f * PY_exp[:, None]).sum())
            lam = max((fac_supply - a_total) / max(b_total, EPS), 0.0)

            PY_eq = np.maximum(PY_cons + lam * PY_exp, EPS)

            # ---- 要素价格更新 ----
            if M > 0:
                w_new = np.array([
                    float((alpha[:, Nl + k] * PY_eq).sum()) / max(L[k], EPS)
                    for k in range(M)
                ])
                if w_new[0] > EPS:
                    w_new = w_new / w_new[0]
                w_change = float(np.max(np.abs(w_new - w_vec)))
                w_vec = 0.5 * w_vec + 0.5 * w_new
                if w_change < 1e-10:
                    break

        # ---- 最终均衡量 ----
        output_eq = np.maximum(PY_eq / np.maximum(P, EPS), EPS)
        exports_eq = lam * E_0

        # 一致性需求
        X_dom = static_intermediate_demand_dom(alpha, theta, price, output_eq, Ml, M)
        X_imp = static_intermediate_demand_imp(
            alpha[:, :Nl], theta, imp_price, price, output_eq, Ml)
        C_dom = static_consumption_demand_dom(beta, theta_c, income, price, Ml)
        C_imp = static_consumption_demand_imp(beta, theta_c, income, imp_price, Ml)

        # P=MC → compute_output 与 Leontief 一致（CES 对偶性）
        output_prod = compute_output(
            A, alpha, X_dom, X_imp, gamma, rho_arr, Ml, M)
        output = np.concatenate([output_prod, L.copy()])

        export_base = np.zeros_like(np.asarray(cp.exports, dtype=float))
        export_base[:Nl] = exports_eq
        export_actual = export_base.copy()

        return CountryState(
            X_dom=X_dom, X_imp=X_imp,
            C_dom=C_dom, C_imp=C_imp,
            price=price, imp_price=imp_price,
            export_base=export_base,
            export_actual=export_actual,
            output=output, income=float(income),
        )

    @staticmethod
    def _block_to_state(block, cp: CountryParams) -> CountryState:
        """从均衡结果 CountryBlock 构造动态状态。"""
        from .types import CountryBlock

        Nl = cp.Nl
        M = cp.M_factors
        output_full = np.concatenate([
            np.asarray(block.output, dtype=float),
            np.asarray(cp.L, dtype=float),
        ])

        # 对于均衡态，actual export ≈ base export
        export_base = np.array(cp.exports, copy=True, dtype=float)
        export_actual = export_base.copy()

        return CountryState(
            X_dom=np.array(block.X_dom, copy=True, dtype=float),
            X_imp=np.array(block.X_imp, copy=True, dtype=float),
            C_dom=np.array(block.C_dom, copy=True, dtype=float),
            C_imp=np.array(block.C_imp, copy=True, dtype=float),
            price=np.array(block.price, copy=True, dtype=float),
            imp_price=np.array(block.imp_price, copy=True, dtype=float),
            export_base=export_base,
            export_actual=export_actual,
            output=output_full,
            income=float(block.income),
        )
