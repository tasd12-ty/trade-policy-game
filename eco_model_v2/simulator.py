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
from .utils import EPS, clamp_positive


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
        normalize_gap: bool = True,
        supply_chain: SupplyChainNetwork | None = None,
    ):
        self._params = params
        self._tau = tau
        self._normalize_gap = normalize_gap
        self.supply_chain = supply_chain

        # 构建动态引擎
        self._rebuild_engines()

        # 历史记录
        self.history: Dict[str, List[CountryState]] = {"H": [], "F": []}

    def _rebuild_engines(self) -> None:
        """重建动态引擎（参数变化后调用）。"""
        self._engine_h = CountryDynamics(
            self._params.home, tau=self._tau,
            normalize_gap=self._normalize_gap,
        )
        self._engine_f = CountryDynamics(
            self._params.foreign, tau=self._tau,
            normalize_gap=self._normalize_gap,
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
        """
        if home_state is None:
            home_state = self._make_initial_state(self._params.home)
        if foreign_state is None:
            foreign_state = self._make_initial_state(self._params.foreign)

        self.history = {"H": [home_state.copy()], "F": [foreign_state.copy()]}

    def initialize_from_equilibrium(self) -> None:
        """从静态均衡初始化（如果求解成功）。"""
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

        修改 import_cost 参数并重建引擎。
        """
        from .presets import apply_tariff

        if country == "H":
            new_home = apply_tariff(self._params.home, tariff_rates)
            self._params = TwoCountryParams(home=new_home, foreign=self._params.foreign)
        else:
            new_foreign = apply_tariff(self._params.foreign, tariff_rates)
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
            "exports": state.export_actual[:Nl].tolist(),
            "opponent_income": opp_state.income,
            "opponent_price": opp_state.price[:Nl].tolist(),
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

    def _make_initial_state(self, cp: CountryParams) -> CountryState:
        """从参数构造简单初始状态。"""
        Nl = cp.Nl
        M = cp.M_factors

        price = np.ones(Nl + M, dtype=float)
        X_dom = np.full((Nl, Nl + M), 0.5, dtype=float)
        X_imp = np.full((Nl, Nl), 0.2, dtype=float)
        X_imp[:, :cp.Ml] = 0.0  # 非贸易品无进口
        C_dom = np.full(Nl, 0.5, dtype=float)
        C_imp = np.full(Nl, 0.2, dtype=float)
        C_imp[:cp.Ml] = 0.0
        imp_price = np.asarray(cp.import_cost, dtype=float) * 1.0
        export_base = np.array(cp.exports, copy=True, dtype=float)
        export_actual = export_base.copy()

        # 产出：从初始投入计算
        output_prod = compute_output(
            cp.A, cp.alpha, X_dom, X_imp,
            cp.gamma, cp.rho, cp.Ml, M,
        )
        output = np.concatenate([output_prod, np.asarray(cp.L, dtype=float)])
        income = compute_income(price, cp.L, Nl)

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
