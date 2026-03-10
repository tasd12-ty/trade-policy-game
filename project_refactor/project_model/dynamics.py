"""动态仿真引擎（解耦版）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .armington import armington_share
from .equilibrium import solve_static_equilibrium
from .production import compute_income, compute_output
from .types import CountryBlock, CountryParams, CountryState, StaticEquilibriumResult, TwoCountryParams
from .utils import EPS


@dataclass(frozen=True)
class CountryDynamicsConfig:
    """单国动态更新参数。"""

    tau_price: float = 0.1
    normalize_gap_by_supply: bool = False


class CountryDynamics:
    """单国离散时间更新器。"""

    def __init__(self, params: CountryParams, tradable_mask: np.ndarray, config: CountryDynamicsConfig):
        self.params = params
        self.tradable_mask = np.asarray(tradable_mask, dtype=bool)
        self.tau = float(config.tau_price)
        self.normalize_gap_by_supply = bool(config.normalize_gap_by_supply)

    def _plan_demands(self, state: CountryState, import_prices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按当期价格与产出计算下一期计划需求。"""
        p = self.params
        outputs = compute_output(p, state.X_dom, state.X_imp, self.tradable_mask)
        income = compute_income(p, state.price, outputs)

        n = p.n
        planned_x_dom = np.zeros((n, n), dtype=float)
        planned_x_imp = np.zeros((n, n), dtype=float)
        planned_c_dom = np.zeros(n, dtype=float)
        planned_c_imp = np.zeros(n, dtype=float)

        # 生产中间品需求
        for i in range(n):
            Pi = float(state.price[i])
            Yi = float(outputs[i])
            for j in range(n):
                a = float(p.alpha[i, j])
                if a <= 0.0:
                    continue

                if self.tradable_mask[j]:
                    theta = float(
                        armington_share(
                            gamma=float(p.gamma[i, j]),
                            p_dom=float(state.price[j]),
                            p_for=float(import_prices[j]),
                            rho=float(p.rho[i, j]),
                        )
                    )
                    planned_x_dom[i, j] = a * theta * Pi * Yi / max(float(state.price[j]), EPS)
                    planned_x_imp[i, j] = a * (1.0 - theta) * Pi * Yi / max(float(import_prices[j]), EPS)
                else:
                    planned_x_dom[i, j] = a * Pi * Yi / max(float(state.price[j]), EPS)

        # 终端消费需求
        for j in range(n):
            b = float(p.beta[j])
            if b <= 0.0:
                continue

            if self.tradable_mask[j]:
                theta_c = float(
                    armington_share(
                        gamma=float(p.gamma_cons[j]),
                        p_dom=float(state.price[j]),
                        p_for=float(import_prices[j]),
                        rho=float(p.rho_cons[j]),
                    )
                )
                planned_c_dom[j] = b * theta_c * income / max(float(state.price[j]), EPS)
                planned_c_imp[j] = b * (1.0 - theta_c) * income / max(float(import_prices[j]), EPS)
            else:
                planned_c_dom[j] = b * income / max(float(state.price[j]), EPS)

        return outputs, income, planned_x_dom, planned_x_imp, planned_c_dom, planned_c_imp

    def _update_prices(
        self,
        state: CountryState,
        outputs: np.ndarray,
        planned_x_dom: np.ndarray,
        planned_c_dom: np.ndarray,
    ) -> np.ndarray:
        """价格更新：P' = P * exp(tau * (D - Y))。"""
        planned_total = planned_x_dom.sum(axis=0) + planned_c_dom + state.export_base
        demand_gap = planned_total - outputs
        if self.normalize_gap_by_supply:
            demand_gap = demand_gap / np.maximum(outputs, EPS)

        delta = self.tau * demand_gap
        # 防止数值爆炸：限制单期变动幅度
        delta = 3.0 * np.tanh(delta / 3.0)
        new_price = state.price * np.exp(delta)
        return np.clip(new_price, EPS, None)

    @staticmethod
    def _allocate_domestic_goods(
        outputs: np.ndarray,
        planned_x_dom: np.ndarray,
        planned_c_dom: np.ndarray,
        planned_export: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """国内供给分配：供不应求时按统一比例缩放。"""
        demand_total = np.maximum(planned_x_dom.sum(axis=0) + planned_c_dom + planned_export, EPS)
        supply = np.maximum(outputs, EPS)
        scale = np.minimum(1.0, supply / demand_total)
        actual_x_dom = planned_x_dom * scale[np.newaxis, :]
        actual_c_dom = planned_c_dom * scale
        actual_export = planned_export * scale
        return actual_x_dom, actual_c_dom, actual_export

    def _allocate_imports_by_fx(
        self,
        import_prices: np.ndarray,
        planned_x_imp: np.ndarray,
        planned_c_imp: np.ndarray,
        export_value: float,
        supply_cap: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """进口外汇约束与可选供给上限。"""
        planned_value = float(np.sum(planned_x_imp * import_prices[np.newaxis, :]) + np.sum(planned_c_imp * import_prices))
        scale_fx = 1.0 if planned_value <= EPS else min(1.0, float(export_value) / planned_value)

        actual_x_imp = planned_x_imp * scale_fx
        actual_c_imp = planned_c_imp * scale_fx

        if supply_cap is not None:
            cap = np.clip(np.asarray(supply_cap, dtype=float), 0.0, None)
            total_imp = actual_x_imp.sum(axis=0) + actual_c_imp
            scale_cap = np.ones_like(total_imp)
            tradable = self.tradable_mask
            tradable_total = np.maximum(total_imp[tradable], EPS)
            scale_cap[tradable] = np.clip(cap[tradable] / tradable_total, 0.0, 1.0)
            actual_x_imp *= scale_cap[np.newaxis, :]
            actual_c_imp *= scale_cap

        return actual_x_imp, actual_c_imp

    def step(
        self,
        state: CountryState,
        import_prices: np.ndarray,
        supply_cap: Optional[np.ndarray] = None,
    ) -> CountryState:
        """执行单期动态更新。"""
        outputs, _, p_x_dom, p_x_imp, p_c_dom, p_c_imp = self._plan_demands(state, import_prices)
        new_price = self._update_prices(state, outputs, p_x_dom, p_c_dom)

        # 采用上一期实际出口作为本期计划出口（与 grad_op 当前策略一致）
        planned_export = np.array(state.export_actual, copy=True)

        a_x_dom, a_c_dom, a_export = self._allocate_domestic_goods(outputs, p_x_dom, p_c_dom, planned_export)
        export_value = float(np.dot(a_export, state.price))
        a_x_imp, a_c_imp = self._allocate_imports_by_fx(import_prices, p_x_imp, p_c_imp, export_value, supply_cap=supply_cap)

        new_output = compute_output(self.params, a_x_dom, a_x_imp, self.tradable_mask)
        new_income = compute_income(self.params, new_price, new_output)

        return CountryState(
            X_dom=a_x_dom,
            X_imp=a_x_imp,
            C_dom=a_c_dom,
            C_imp=a_c_imp,
            price=new_price,
            imp_price=np.asarray(import_prices, dtype=float),
            export_base=np.array(state.export_base, copy=True),
            export_actual=a_export,
            output=new_output,
            income=float(new_income),
        )


def _extract_export_base(partner_block: CountryBlock) -> np.ndarray:
    """由对方进口需求反推本国基准出口。"""
    return np.asarray(partner_block.X_imp.sum(axis=0) + partner_block.C_imp, dtype=float)


def _build_state(block: CountryBlock, params: CountryParams, export_base: np.ndarray, tradable_mask: np.ndarray) -> CountryState:
    output = compute_output(params, block.X_dom, block.X_imp, tradable_mask)
    income = compute_income(params, block.price, output)
    return CountryState(
        X_dom=np.array(block.X_dom, copy=True),
        X_imp=np.array(block.X_imp, copy=True),
        C_dom=np.array(block.C_dom, copy=True),
        C_imp=np.array(block.C_imp, copy=True),
        price=np.array(block.price, copy=True),
        imp_price=np.array(block.imp_price, copy=True),
        export_base=np.array(export_base, copy=True),
        export_actual=np.array(export_base, copy=True),
        output=np.array(output, copy=True),
        income=float(income),
    )


class TwoCountrySimulator:
    """两国动态仿真器。"""

    def __init__(
        self,
        params: TwoCountryParams,
        equilibrium: StaticEquilibriumResult,
        tau_price: float = 0.1,
        normalize_gap_by_supply: bool = False,
    ):
        self.params = params
        self.tradable_mask = params.tradable_mask

        export_h = _extract_export_base(equilibrium.foreign)
        export_f = _extract_export_base(equilibrium.home)

        self.home_state = _build_state(equilibrium.home, params.home, export_h, self.tradable_mask)
        self.foreign_state = _build_state(equilibrium.foreign, params.foreign, export_f, self.tradable_mask)

        cfg = CountryDynamicsConfig(tau_price=tau_price, normalize_gap_by_supply=normalize_gap_by_supply)
        self.home_dyn = CountryDynamics(params.home, self.tradable_mask, cfg)
        self.foreign_dyn = CountryDynamics(params.foreign, self.tradable_mask, cfg)

        self.history: Dict[str, List[CountryState]] = {
            "H": [self.home_state.copy()],
            "F": [self.foreign_state.copy()],
        }

        self.baseline_export = {
            "H": np.array(self.home_state.export_base, copy=True),
            "F": np.array(self.foreign_state.export_base, copy=True),
        }
        self.export_multiplier = {
            "H": np.ones(params.n, dtype=float),
            "F": np.ones(params.n, dtype=float),
        }

        self.home_import_multiplier = np.array(params.home.import_cost, copy=True)
        self.foreign_import_multiplier = np.array(params.foreign.import_cost, copy=True)
        self.baseline_import_multiplier = {
            "H": np.array(self.home_import_multiplier, copy=True),
            "F": np.array(self.foreign_import_multiplier, copy=True),
        }

        # 进口供给上限系数：默认可贸易部门为 1，不可贸易为 0
        tradable_f = self.tradable_mask.astype(float)
        self.import_cap_coeff = {
            "H": np.array(tradable_f, copy=True),
            "F": np.array(tradable_f, copy=True),
        }

        self.policy_events: List[Dict[str, Any]] = []

        p_h, p_f = self._import_prices()
        self.home_state.imp_price = p_h
        self.foreign_state.imp_price = p_f

    def _current_period(self) -> int:
        return len(self.history["H"]) - 1

    def _log_event(self, event: Dict[str, Any]) -> None:
        event.setdefault("period", self._current_period())
        self.policy_events.append(event)

    def _import_prices(self) -> Tuple[np.ndarray, np.ndarray]:
        p_h = self.home_import_multiplier * self.foreign_state.price
        p_f = self.foreign_import_multiplier * self.home_state.price
        return p_h, p_f

    def _update_export_base(self, country: str) -> None:
        c = country.upper()
        updated = np.clip(self.baseline_export[c] * self.export_multiplier[c], 0.0, None)
        if c == "H":
            self.home_state.export_base = updated
        else:
            self.foreign_state.export_base = updated

    def step(self) -> None:
        p_h, p_f = self._import_prices()

        cap_h = self.import_cap_coeff["H"] * self.foreign_state.export_actual
        cap_f = self.import_cap_coeff["F"] * self.home_state.export_actual

        new_h = self.home_dyn.step(self.home_state, p_h, supply_cap=cap_h)
        new_f = self.foreign_dyn.step(self.foreign_state, p_f, supply_cap=cap_f)

        self.home_state = new_h
        self.foreign_state = new_f
        self.history["H"].append(new_h.copy())
        self.history["F"].append(new_f.copy())

    def run(self, periods: int) -> None:
        for _ in range(int(periods)):
            self.step()

    def apply_export_control(self, country: str, sector_factors: Dict[int, float], note: Optional[str] = None) -> None:
        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        sanitized: Dict[int, float] = {}
        for sec, fac in sector_factors.items():
            s = int(sec)
            if s < 0 or s >= self.params.n:
                raise IndexError(f"部门索引越界: {s}")
            val = max(float(fac), 0.0)
            self.export_multiplier[c][s] = val
            sanitized[s] = val
        self._update_export_base(c)

        event = {"type": "export_control", "country": c, "sectors": sanitized}
        if note:
            event["note"] = note
        self._log_event(event)

    def reset_export_control(self, country: str, sectors: Optional[List[int]] = None, note: Optional[str] = None) -> None:
        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        if sectors is None:
            self.export_multiplier[c][:] = 1.0
            touched = list(range(self.params.n))
        else:
            touched = []
            for sec in sectors:
                s = int(sec)
                if s < 0 or s >= self.params.n:
                    raise IndexError(f"部门索引越界: {s}")
                self.export_multiplier[c][s] = 1.0
                touched.append(s)
        self._update_export_base(c)

        event = {"type": "export_control_reset", "country": c, "sectors": touched}
        if note:
            event["note"] = note
        self._log_event(event)

    def apply_import_tariff(self, country: str, sector_rates: Dict[int, float], note: Optional[str] = None) -> None:
        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        baseline = self.baseline_import_multiplier[c]
        target = self.home_import_multiplier if c == "H" else self.foreign_import_multiplier

        sanitized: Dict[int, float] = {}
        for sec, rate in sector_rates.items():
            s = int(sec)
            if s < 0 or s >= self.params.n:
                raise IndexError(f"部门索引越界: {s}")
            r = float(rate)
            target[s] = max(float(baseline[s] * (1.0 + r)), 0.0)
            sanitized[s] = r

        event = {"type": "import_tariff", "country": c, "sectors": sanitized}
        if note:
            event["note"] = note
        self._log_event(event)

    def set_import_multiplier(
        self,
        country: str,
        sector_multipliers: Dict[int, float],
        *,
        relative_to_baseline: bool = True,
        note: Optional[str] = None,
    ) -> None:
        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        baseline = self.baseline_import_multiplier[c]
        target = self.home_import_multiplier if c == "H" else self.foreign_import_multiplier

        sanitized: Dict[int, float] = {}
        for sec, m in sector_multipliers.items():
            s = int(sec)
            if s < 0 or s >= self.params.n:
                raise IndexError(f"部门索引越界: {s}")
            mul = float(m)
            if relative_to_baseline:
                target[s] = max(float(baseline[s] * mul), 0.0)
            else:
                target[s] = max(mul, 0.0)
            sanitized[s] = mul

        event = {
            "type": "import_multiplier",
            "country": c,
            "relative": bool(relative_to_baseline),
            "sectors": sanitized,
        }
        if note:
            event["note"] = note
        self._log_event(event)

    def reset_import_policies(self, country: str, sectors: Optional[List[int]] = None, note: Optional[str] = None) -> None:
        c = country.upper()
        if c not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        baseline = self.baseline_import_multiplier[c]
        target = self.home_import_multiplier if c == "H" else self.foreign_import_multiplier

        if sectors is None:
            target[:] = baseline
            touched = list(range(self.params.n))
        else:
            touched = []
            for sec in sectors:
                s = int(sec)
                if s < 0 or s >= self.params.n:
                    raise IndexError(f"部门索引越界: {s}")
                target[s] = baseline[s]
                touched.append(s)

        event = {"type": "import_policy_reset", "country": c, "sectors": touched}
        if note:
            event["note"] = note
        self._log_event(event)

    def summarize_history(self) -> Dict[str, Dict[str, np.ndarray]]:
        """输出简要时序指标。"""

        def summarize(states: List[CountryState]) -> Dict[str, np.ndarray]:
            price_mean = np.asarray([float(np.mean(s.price)) for s in states], dtype=float)
            output_sum = np.asarray([float(np.sum(s.output)) for s in states], dtype=float)
            income = np.asarray([float(s.income) for s in states], dtype=float)
            export_value = np.asarray([float(np.dot(s.export_actual, s.price)) for s in states], dtype=float)
            import_value = np.asarray(
                [float(np.dot(s.imp_price, s.X_imp.sum(axis=0) + s.C_imp)) for s in states],
                dtype=float,
            )
            trade_balance = export_value - import_value
            return {
                "price_mean": price_mean,
                "output_sum": output_sum,
                "income": income,
                "export_value": export_value,
                "import_value": import_value,
                "trade_balance": trade_balance,
            }

        return {
            "H": summarize(self.history["H"]),
            "F": summarize(self.history["F"]),
        }


def bootstrap_dynamic_simulator(
    params: TwoCountryParams,
    tau_price: float = 0.1,
    normalize_gap_by_supply: bool = False,
    max_iterations: int = 400,
    tolerance: float = 1e-8,
) -> TwoCountrySimulator:
    """静态均衡 + 动态仿真器一键启动。"""
    eq = solve_static_equilibrium(params, max_iterations=max_iterations, tolerance=tolerance)
    return TwoCountrySimulator(params, eq, tau_price=tau_price, normalize_gap_by_supply=normalize_gap_by_supply)
