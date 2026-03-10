r"""动态仿真（two-country, multi-sector；NumPy）。

使用 production_network_simulation0916.tex 的迭代方程：
- 需求调整：\Delta x_D, \Delta c_D（对数变化）
- 价格更新：\Delta p_t = Diag(\tau) * (D - Y)
- 实际分配：按供给比例缩放（式 17-18）
- 外汇约束：按出口收入缩放进口（式 20-21）

该实现不依赖梯度与 PyTorch。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import copy
import numpy as np

from .model import (
    CountryParams,
    ModelParams,
    create_symmetric_parameters,
    normalize_model_params,
    EPS,
    safe_log,
    compute_output,
    compute_marginal_cost,
    compute_income,
    solve_initial_equilibrium,
)


# ---------------------
# 状态表示
# ---------------------


@dataclass
class CountryState:
    """单国状态快照。"""

    X_dom: np.ndarray
    X_imp: np.ndarray
    C_dom: np.ndarray
    C_imp: np.ndarray
    price: np.ndarray
    imp_price: np.ndarray
    export_base: np.ndarray
    export_actual: np.ndarray
    output: np.ndarray
    income: float

    def copy(self) -> "CountryState":
        return CountryState(
            X_dom=self.X_dom.copy(),
            X_imp=self.X_imp.copy(),
            C_dom=self.C_dom.copy(),
            C_imp=self.C_imp.copy(),
            price=self.price.copy(),
            imp_price=self.imp_price.copy(),
            export_base=self.export_base.copy(),
            export_actual=self.export_actual.copy(),
            output=self.output.copy(),
            income=float(self.income),
        )


# ---------------------
# 单国动态更新器
# ---------------------


class CountrySimulator:
    """单国的计划—价格—分配—再计算一周期更新器。"""

    def __init__(
        self,
        params: CountryParams,
        tradable_mask: np.ndarray,
        theta_price: float = 12500,
        *,
        normalize_gap_by_supply: bool = False,
    ):
        self.params = params
        self.tradable_mask = tradable_mask
        self.tau = np.full((params.alpha.shape[0],), float(theta_price))
        self.normalize_gap_by_supply = bool(normalize_gap_by_supply)

    @staticmethod
    def _theta_from_usage(gamma: np.ndarray, rho: np.ndarray,
                          X_dom: np.ndarray, X_imp: np.ndarray) -> np.ndarray:
        g = np.clip(gamma, 1e-6, 1.0 - 1e-6)
        r = rho
        clamped_dom = np.clip(X_dom, EPS, None)
        clamped_imp = np.clip(X_imp, EPS, None)
        dom = g * (clamped_dom ** r)
        imp = (1.0 - g) * (clamped_imp ** r)
        denom = np.clip(dom + imp, EPS, None)
        theta = dom / denom
        theta = np.where(np.abs(r) < 1e-10, g, theta)
        return np.clip(theta, 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _theta_consumption(gamma_c: np.ndarray, rho_c: np.ndarray,
                           C_dom: np.ndarray, C_imp: np.ndarray) -> np.ndarray:
        g = np.clip(gamma_c, 1e-6, 1.0 - 1e-6)
        r = rho_c
        c_dom = np.clip(C_dom, EPS, None)
        c_imp = np.clip(C_imp, EPS, None)
        dom = g * (c_dom ** r)
        imp = (1.0 - g) * (c_imp ** r)
        denom = np.clip(dom + imp, EPS, None)
        theta = dom / denom
        theta = np.where(np.abs(r) < 1e-10, g, theta)
        return np.clip(theta, 1e-6, 1.0 - 1e-6)

    def _plan_demands(self, state: CountryState, import_prices: np.ndarray) -> Tuple[np.ndarray, ...]:
        p = self.params
        outputs = compute_output(p, state.X_dom, state.X_imp, self.tradable_mask)
        lambdas = compute_marginal_cost(p, state.price, import_prices, self.tradable_mask)
        theta_prod_full = CountrySimulator._theta_from_usage(p.gamma, p.rho, state.X_dom, state.X_imp)
        theta_cons_full = CountrySimulator._theta_consumption(p.gamma_cons, p.rho_cons, state.C_dom, state.C_imp)
        theta_prod = np.where(self.tradable_mask[None, :], theta_prod_full, 1.0)
        theta_cons = np.where(self.tradable_mask, theta_cons_full, 1.0)

        mask_a = p.alpha > 0
        log_alpha = np.where(mask_a, safe_log(p.alpha), 0.0)
        log_lambda = safe_log(lambdas)[:, None]
        log_Y = safe_log(outputs)[:, None]
        log_price = safe_log(state.price)[None, :]
        log_import_price = safe_log(import_prices)[None, :]
        log_X_dom = safe_log(state.X_dom)
        log_X_imp = safe_log(state.X_imp)

        log_theta_prod = safe_log(theta_prod)
        log_one_minus_theta_prod = safe_log(1.0 - theta_prod)

        delta_dom = np.zeros_like(state.X_dom)
        delta_imp = np.zeros_like(state.X_imp)

        trad_mask = self.tradable_mask[None, :]
        delta_dom_trad = (log_alpha + log_lambda + log_theta_prod + log_Y - log_price - log_X_dom)
        delta_imp_trad = (log_alpha + log_lambda + log_one_minus_theta_prod + log_Y - log_import_price - log_X_imp)
        delta_dom = np.where(mask_a & trad_mask, delta_dom_trad, delta_dom)
        delta_imp = np.where(mask_a & trad_mask, delta_imp_trad, delta_imp)

        non_trad_mask = (~self.tradable_mask)[None, :]
        delta_dom_non = (log_alpha + log_lambda + log_Y - log_price - log_X_dom)
        delta_dom = np.where(mask_a & non_trad_mask, delta_dom_non, delta_dom)

        mask_b = p.beta > 0
        log_b = np.where(mask_b, safe_log(p.beta), 0.0)
        log_income = safe_log(state.income)
        log_price_vec = safe_log(state.price)
        log_import_price_vec = safe_log(import_prices)
        log_C_dom = safe_log(state.C_dom)
        log_C_imp = safe_log(state.C_imp)
        log_theta_cons = safe_log(theta_cons)
        log_one_minus_theta_cons = safe_log(1.0 - theta_cons)

        delta_c_dom = np.zeros_like(state.C_dom)
        delta_c_imp = np.zeros_like(state.C_imp)

        delta_c_dom_trad = log_b + log_income + log_theta_cons - log_price_vec - log_C_dom
        delta_c_imp_trad = log_b + log_income + log_one_minus_theta_cons - log_import_price_vec - log_C_imp
        delta_c_dom = np.where(mask_b & self.tradable_mask, delta_c_dom_trad, delta_c_dom)
        delta_c_imp = np.where(mask_b & self.tradable_mask, delta_c_imp_trad, delta_c_imp)

        delta_c_dom_non = log_b + log_income - log_price_vec - log_C_dom
        delta_c_dom = np.where(mask_b & (~self.tradable_mask), delta_c_dom_non, delta_c_dom)

        planned_X_dom = state.X_dom * np.exp(delta_dom)
        planned_X_imp = state.X_imp * np.exp(delta_imp)
        planned_C_dom = state.C_dom * np.exp(delta_c_dom)
        planned_C_imp = state.C_imp * np.exp(delta_c_imp)
        return outputs, planned_X_dom, planned_X_imp, planned_C_dom, planned_C_imp, theta_prod, theta_cons

    def _update_prices(self, state: CountryState, outputs: np.ndarray,
                       planned_X_dom: np.ndarray, planned_C_dom: np.ndarray) -> np.ndarray:
        planned_total = planned_X_dom.sum(axis=0) + planned_C_dom + state.export_base
        demand_gap = planned_total - outputs
        if self.normalize_gap_by_supply:
            supply = np.clip(outputs, EPS, None)
            demand_gap = demand_gap / supply
        delta_demand = self.tau * demand_gap
        return state.price * np.exp(delta_demand)

    def _allocate_goods(self, outputs: np.ndarray, planned_X_dom: np.ndarray,
                        planned_C_dom: np.ndarray, planned_export: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        demand_total = planned_X_dom.sum(axis=0) + planned_C_dom + planned_export
        supply = outputs
        ratio = supply / np.clip(demand_total, EPS, None)
        scale = np.minimum(1.0, ratio)
        actual_X_dom = planned_X_dom * scale[None, :]
        actual_C_dom = planned_C_dom * scale
        actual_export = planned_export * scale
        return actual_X_dom, actual_C_dom, actual_export

    def _allocate_imports_fx(self, import_prices: np.ndarray, planned_X_imp: np.ndarray,
                             planned_C_imp: np.ndarray, export_value: float,
                             supply_cap: Optional[np.ndarray] = None,
                             tradable_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        planned_value = float((planned_X_imp * import_prices[None, :]).sum() + (planned_C_imp * import_prices).sum())
        if planned_value > EPS:
            scale_fx = min(1.0, export_value / planned_value)
        else:
            scale_fx = 1.0
        sX = planned_X_imp * scale_fx
        sC = planned_C_imp * scale_fx

        if supply_cap is not None:
            mask_vec = tradable_mask if tradable_mask is not None else np.ones(sC.size, dtype=bool)
            total = sX.sum(axis=0) + sC
            cap = np.clip(supply_cap, 0.0, None)
            scale_cap = np.clip(cap / np.clip(total, EPS, None), 0.0, 1.0)
            scale_cap = np.where(mask_vec, scale_cap, 1.0)
            sX = sX * scale_cap[None, :]
            sC = sC * scale_cap
        return sX, sC

    def step(self, state: CountryState, import_prices: np.ndarray,
             supply_cap: Optional[np.ndarray] = None,
             tradable_mask: Optional[np.ndarray] = None) -> CountryState:
        outputs, pX_dom, pX_imp, pC_dom, pC_imp, _, _ = self._plan_demands(state, import_prices)
        new_price = self._update_prices(state, outputs, pX_dom, pC_dom)

        planned_export = state.export_base.copy()
        aX_dom, aC_dom, aExport = self._allocate_goods(outputs, pX_dom, pC_dom, planned_export)
        export_value = float((aExport * state.price).sum())
        aX_imp, aC_imp = self._allocate_imports_fx(import_prices, pX_imp, pC_imp, export_value,
                                                   supply_cap=supply_cap, tradable_mask=tradable_mask)
        new_output = compute_output(self.params, aX_dom, aX_imp, self.tradable_mask)
        new_income = compute_income(self.params, new_price, new_output)
        return CountryState(aX_dom, aX_imp, aC_dom, aC_imp, new_price, import_prices,
                            state.export_base, aExport, new_output, new_income)


# ---------------------
# 两国仿真器
# ---------------------


def _merge_consumption(block: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    c_dom = np.asarray(block["C_j"], dtype=float)
    c_dom = c_dom + np.asarray(block["C_I_j"], dtype=float)
    c_imp = np.asarray(block["C_O_j"], dtype=float)
    return c_dom, c_imp


def _extract_export_base(partner_block: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    interm = np.asarray(partner_block["intermediate_inputs"]["X_O_ij"], dtype=float)
    consumption = np.asarray(partner_block["final_consumption"]["C_O_j"], dtype=float)
    return np.sum(interm, axis=0) + consumption


def _build_state(block: Dict[str, Dict[str, np.ndarray]], params: CountryParams,
                 export_base: np.ndarray, tradable_mask: np.ndarray) -> CountryState:
    X_dom = np.asarray(block["intermediate_inputs"]["X_ij"], dtype=float)
    X_imp = np.asarray(block["intermediate_inputs"]["X_O_ij"], dtype=float)
    C_dom, C_imp = _merge_consumption(block["final_consumption"])
    prices = np.asarray(block["prices"]["P_j"], dtype=float)
    output = compute_output(params, X_dom, X_imp, tradable_mask)
    income = compute_income(params, prices, output)
    imp_price = np.ones_like(prices)
    return CountryState(X_dom, X_imp, C_dom, C_imp, prices, imp_price,
                        export_base, export_base.copy(), output, income)


class TwoCountryDynamicSimulator:
    """两国动态仿真器（围绕静态均衡的政策冲击与响应路径）。"""

    def __init__(
        self,
        params: ModelParams,
        equilibrium: Dict[str, Dict],
        theta_price: float = 12500,
        *,
        normalize_gap_by_supply: bool = False,
    ):
        self.params = params
        tmask = np.zeros(params.home.alpha.shape[0], bool)
        tmask[params.tradable_idx] = True
        self.tradable_mask = tmask

        export_H = _extract_export_base(equilibrium["F"])
        export_F = _extract_export_base(equilibrium["H"])

        self.home_state = _build_state(equilibrium["H"], params.home, export_H, tmask)
        self.foreign_state = _build_state(equilibrium["F"], params.foreign, export_F, tmask)

        self.home_sim = CountrySimulator(params.home, tmask, theta_price, normalize_gap_by_supply=normalize_gap_by_supply)
        self.foreign_sim = CountrySimulator(params.foreign, tmask, theta_price, normalize_gap_by_supply=normalize_gap_by_supply)

        self.history: Dict[str, List[CountryState]] = {"H": [self.home_state], "F": [self.foreign_state]}

        self.baseline_export = {"H": self.home_state.export_base.copy(), "F": self.foreign_state.export_base.copy()}
        self.export_multiplier = {"H": np.ones_like(self.baseline_export["H"]), "F": np.ones_like(self.baseline_export["F"])}

        self.home_import_multiplier = params.home.import_cost.copy()
        self.foreign_import_multiplier = params.foreign.import_cost.copy()
        self.baseline_import_multiplier = {"H": self.home_import_multiplier.copy(), "F": self.foreign_import_multiplier.copy()}

        n = self.baseline_export["H"].shape[0]
        mask_values = tmask.astype(float)
        self.import_cap_coeff: Dict[str, np.ndarray] = {"H": mask_values.copy(), "F": mask_values.copy()}

        self.policy_events: List[Dict[str, Any]] = []

        pH, pF = self._import_prices()
        self.home_state.imp_price = pH.copy()
        self.foreign_state.imp_price = pF.copy()

    def _import_prices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.home_import_multiplier * self.foreign_state.price, self.foreign_import_multiplier * self.home_state.price

    def step(self) -> None:
        pH, pF = self._import_prices()
        cap_H = self.import_cap_coeff["H"] * self.foreign_state.export_actual
        cap_F = self.import_cap_coeff["F"] * self.home_state.export_actual
        new_H = self.home_sim.step(self.home_state, pH, supply_cap=cap_H, tradable_mask=self.tradable_mask)
        new_F = self.foreign_sim.step(self.foreign_state, pF, supply_cap=cap_F, tradable_mask=self.tradable_mask)
        self.home_state, self.foreign_state = new_H, new_F
        self.history["H"].append(new_H.copy())
        self.history["F"].append(new_F.copy())

    def run(self, periods: int) -> None:
        for _ in range(periods):
            self.step()

    def summarize_history(self) -> Dict[str, Dict[str, np.ndarray]]:
        def extract(states: List[CountryState]) -> Dict[str, np.ndarray]:
            price_mean = np.array([float(s.price.mean()) for s in states], float)
            output_sum = np.array([float(s.output.sum()) for s in states], float)
            income = np.array([float(s.income) for s in states], float)
            export_value = np.array([float((s.export_actual * s.price).sum()) for s in states], float)
            import_qty = np.array([float(s.C_imp.sum() + s.X_imp.sum()) for s in states], float)
            trade_balance = export_value - import_qty
            import_value_val = np.array([float((s.imp_price * (s.X_imp.sum(axis=0) + s.C_imp)).sum()) for s in states], float)
            trade_balance_val = export_value - import_value_val
            income_growth = ((income / income[0]) - 1) * 100 if income[0] > 0 else np.zeros_like(income)
            output_growth = ((output_sum / output_sum[0]) - 1) * 100 if output_sum[0] > 0 else np.zeros_like(output_sum)
            return {
                "price_mean": price_mean,
                "output_sum": output_sum,
                "income": income,
                "export_value": export_value,
                "import_value": import_qty,
                "trade_balance": trade_balance,
                "import_value_val": import_value_val,
                "trade_balance_val": trade_balance_val,
                "income_growth": income_growth,
                "output_growth": output_growth,
            }

        return {"H": extract(self.history["H"]), "F": extract(self.history["F"])}

    def get_detailed_history(
        self,
        country: str,
        start_period: Optional[int] = None,
        end_period: Optional[int] = None,
    ) -> List["PeriodRecord"]:
        from .sector_history import build_period_record

        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")

        states = self.history[country]
        n_sectors = int(self.params.home.alpha.shape[0])

        def _get_tariff_rates(c: str) -> Dict[int, float]:
            baseline = self.baseline_import_multiplier[c]
            current = self.home_import_multiplier if c == "H" else self.foreign_import_multiplier
            rates = {}
            for j in range(n_sectors):
                base_val = float(baseline[j])
                curr_val = float(current[j])
                if base_val > 1e-6:
                    rates[j] = (curr_val / base_val) - 1.0
                else:
                    rates[j] = 0.0
            return rates

        def _get_quota_rates(c: str) -> Dict[int, float]:
            mult = self.export_multiplier[c]
            return {j: float(mult[j]) for j in range(n_sectors)}

        start = start_period if start_period is not None else 0
        end = end_period if end_period is not None else len(states)

        tariff_rates = _get_tariff_rates(country)
        quota_rates = _get_quota_rates(country)

        records = []
        for t in range(start, min(end, len(states))):
            record = build_period_record(
                state=states[t],
                country=country,
                period=t,
                n_sectors=n_sectors,
                tariff_rates=tariff_rates,
                quota_rates=quota_rates,
            )
            records.append(record)

        return records

    def get_recent_history_summary(self, country: str, num_periods: int = 5) -> str:
        total = len(self.history[country.upper()])
        start = max(0, total - num_periods)
        records = self.get_detailed_history(country, start_period=start)

        if not records:
            return "暂无历史数据"

        lines = [f"### {country.upper()} 国最近 {len(records)} 期数据"]
        for rec in records:
            lines.append(rec.summary_str(max_sectors=None))

        return "\n".join(lines)

    def plot_history(self, *args, **kwargs):
        from .plotting import plot_history as _ph
        return _ph(self, *args, **kwargs)

    def plot_sector_paths(self, *args, **kwargs):
        from .plotting import plot_sector_paths as _ps
        return _ps(self, *args, **kwargs)

    def plot_diagnostics(self, *args, **kwargs):
        from .plotting import plot_diagnostics as _pd
        return _pd(self, *args, **kwargs)

    def _current_period(self) -> int:
        return len(self.history["H"]) - 1

    def _log_policy_event(self, event: Dict[str, Any]) -> None:
        event.setdefault("period", self._current_period())
        self.policy_events.append(event)

    def _update_export_base(self, country: str) -> None:
        baseline = self.baseline_export[country]
        mult = self.export_multiplier[country]
        updated = np.clip(baseline * mult, 0.0, None)
        target_state = self.home_state if country == "H" else self.foreign_state
        target_state.export_base = updated.copy()

    def apply_export_control(self, country: str, sector_factors: Dict[int, float], note: Optional[str] = None) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        mult = self.export_multiplier[country]
        size = mult.size
        sanitized: Dict[int, float] = {}
        for sec, fac in sector_factors.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            val = float(max(fac, 0.0))
            mult[sec] = val
            sanitized[int(sec)] = val
        self._update_export_base(country)
        ev = {"type": "export_control", "country": country, "sectors": sanitized}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def reset_export_control(self, country: str, sectors: Optional[List[int]] = None, note: Optional[str] = None) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        mult = self.export_multiplier[country]
        if sectors is None:
            mult[:] = 1.0
            touched = list(range(mult.size))
        else:
            touched = []
            for sec in sectors:
                if not 0 <= sec < mult.size:
                    raise IndexError(f"部门索引 {sec} 超出范围 0-{mult.size - 1}")
                mult[sec] = 1.0
                touched.append(int(sec))
        self._update_export_base(country)
        ev = {"type": "export_control_reset", "country": country, "sectors": touched}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def _import_multiplier_array(self, country: str) -> np.ndarray:
        return self.home_import_multiplier if country == "H" else self.foreign_import_multiplier

    def apply_import_tariff(self, country: str, sector_rates: Dict[int, float], note: Optional[str] = None) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size
        sanitized: Dict[int, float] = {}
        for sec, rate in sector_rates.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            target = baseline[sec] * (1.0 + float(rate))
            mult[sec] = max(target, 0.0)
            sanitized[int(sec)] = float(rate)
        ev = {"type": "import_tariff", "country": country, "sectors": sanitized}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def set_import_multiplier(self, country: str, sector_multipliers: Dict[int, float], *,
                               relative_to_baseline: bool = True, note: Optional[str] = None) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size
        sanitized: Dict[int, float] = {}
        for sec, m in sector_multipliers.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            target = baseline[sec] * float(m) if relative_to_baseline else float(m)
            mult[sec] = max(target, 0.0)
            sanitized[int(sec)] = float(m)
        ev = {"type": "import_multiplier", "country": country, "relative": relative_to_baseline, "sectors": sanitized}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def apply_action(self, actor: str, action: Dict[str, Any]) -> None:
        actor = actor.upper()

        tariff = action.get("import_tariff")
        if tariff:
            self.apply_import_tariff(actor, tariff)

        quota = action.get("export_quota")
        if quota:
            self.apply_export_control(actor, quota)

        multiplier = action.get("import_multiplier")
        if multiplier:
            self.set_import_multiplier(actor, multiplier, relative_to_baseline=True)

    def reset_import_policies(self, country: str, sectors: Optional[List[int]] = None, note: Optional[str] = None) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size
        if sectors is None:
            mult[:] = baseline
            touched = list(range(size))
        else:
            touched = []
            for sec in sectors:
                if not 0 <= sec < size:
                    raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
                mult[sec] = baseline[sec]
                touched.append(int(sec))
        ev = {"type": "import_policy_reset", "country": country, "sectors": touched}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def set_import_supply_cap_coeff(self, country: str, sector_coeffs: Dict[int, float]) -> None:
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        arr = self.import_cap_coeff[country]
        size = arr.size
        for sec, val in sector_coeffs.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            arr[sec] = float(max(val, 0.0)) if self.tradable_mask[sec] else 0.0

    def clone(self) -> "TwoCountryDynamicSimulator":
        return copy.deepcopy(self)

    def fork(self, keep_history: bool = False) -> "TwoCountryDynamicSimulator":
        if keep_history:
            return self.clone()
        new = self.clone()
        new.history = {"H": [new.home_state.copy()], "F": [new.foreign_state.copy()]}
        return new


# ---------------------
# 时间线与仿真入口
# ---------------------


@dataclass
class PolicyEvent:
    """标准化政策事件：时间区间 [start, end)，与扮演者、部门映射。"""
    kind: str
    actor: str
    sectors: Dict[int, float]
    start_period: int
    end_period: Optional[int] = None
    note: Optional[str] = None


@dataclass
class ConflictBlock:
    """冲突块：高层语法糖，自动展开为一组 PolicyEvent。"""
    export_controls: Dict[str, Dict[int, float]] = field(default_factory=dict)
    import_tariffs: Dict[str, Dict[int, float]] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """仿真配置：总期数、冲突起点、求解器精度与事件列表等。"""
    total_periods: int
    conflict_start: int
    theta_price: float = 12500
    normalize_gap_by_supply: bool = False
    solver_max_iter: int = 400
    solver_tol: float = 1e-8
    events: Optional[List[PolicyEvent]] = None
    conflict: Optional[ConflictBlock] = None


def _events_from_conflict(block: ConflictBlock, start: int) -> List[PolicyEvent]:
    evs: List[PolicyEvent] = []
    for actor, mapping in (block.export_controls or {}).items():
        evs.append(PolicyEvent(kind="export_quota", actor=actor, sectors=dict(mapping), start_period=start))
    for actor, mapping in (block.import_tariffs or {}).items():
        evs.append(PolicyEvent(kind="import_tariff", actor=actor, sectors=dict(mapping), start_period=start))
    return evs


def bootstrap_simulator(
    params_raw: Dict[str, Dict],
    theta_price: float = 12500,
    *,
    normalize_gap_by_supply: bool = False,
) -> TwoCountryDynamicSimulator:
    """便捷启动：以默认精度求解静态均衡并构造仿真器。"""
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params, max_iterations=400, tolerance=1e-8)
    return TwoCountryDynamicSimulator(params, eqm, theta_price, normalize_gap_by_supply=normalize_gap_by_supply)


def simulate(config: SimulationConfig, params_raw: Optional[Dict[str, Dict]] = None) -> TwoCountryDynamicSimulator:
    """运行带时间线的两国仿真：按 config 生成/合并 PolicyEvent，逐期施加并推进。"""
    if params_raw is None:
        params_raw = create_symmetric_parameters()
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params, max_iterations=config.solver_max_iter, tolerance=config.solver_tol)
    sim = TwoCountryDynamicSimulator(
        params,
        eqm,
        theta_price=config.theta_price,
        normalize_gap_by_supply=bool(getattr(config, "normalize_gap_by_supply", False)),
    )

    timeline: List[PolicyEvent] = []
    if config.conflict is not None:
        timeline.extend(_events_from_conflict(config.conflict, config.conflict_start))
    if config.events is not None:
        timeline.extend(list(config.events))

    start_bucket: Dict[int, List[PolicyEvent]] = {}
    end_bucket: Dict[int, List[PolicyEvent]] = {}
    for ev in timeline:
        start_bucket.setdefault(int(ev.start_period), []).append(ev)
        if ev.end_period is not None:
            end_bucket.setdefault(int(ev.end_period), []).append(ev)

    total = int(config.total_periods)
    for t in range(total):
        if t in end_bucket:
            for ev in end_bucket[t]:
                if ev.kind == "export_quota":
                    sim.reset_export_control(ev.actor, sectors=list(ev.sectors.keys()), note=f"结束期 {t} 还原出口限额")
                elif ev.kind in ("import_tariff", "import_multiplier"):
                    sim.reset_import_policies(ev.actor, sectors=list(ev.sectors.keys()), note=f"结束期 {t} 还原进口政策")

        if t in start_bucket:
            for ev in start_bucket[t]:
                if ev.kind == "export_quota":
                    sim.apply_export_control(ev.actor, ev.sectors, note=ev.note or f"期 {t} 出口限额")
                elif ev.kind == "import_tariff":
                    sim.apply_import_tariff(ev.actor, ev.sectors, note=ev.note or f"期 {t} 进口关税")
                elif ev.kind == "import_multiplier":
                    sim.set_import_multiplier(ev.actor, ev.sectors, relative_to_baseline=True, note=ev.note or f"期 {t} 进口乘子")

        sim.step()

    return sim


__all__ = [
    "CountryState",
    "CountrySimulator",
    "TwoCountryDynamicSimulator",
    "PolicyEvent",
    "ConflictBlock",
    "SimulationConfig",
    "bootstrap_simulator",
    "simulate",
]
