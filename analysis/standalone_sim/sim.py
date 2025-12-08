"""动态仿真（two-country, multi-sector；PyTorch 张量）。

目标：在静态均衡附近引入价格—需求—供给的离散时间调整，配合政策冲击与约束，生成路径。

核心规则（简化、可替换）：
- 价格更新（逐部门）：
    P_{t+1} = P_t · exp(τ · (D_t - Y_t))，其中 D_t 为计划内的总使用（中间+消费+出口基线），
    τ 为价格调整速度。若供不应求（D>Y），价格上升，反之下降。
- 国内供给分配：
    若 D_total > Y，则等比例缩放各用途（中间/消费/出口基线）的实际成交数量；
    缩放系数 s = min(1, Y / D_total)。
- 进口外汇约束：
    令计划进口值 V_imp = Σ_j (P_j^O·X^O_{·j}) + Σ_j (P_j^O·C^O_j)，出口值 V_exp = Σ_j (P_j·Export_j^actual)。
    以 s_fx = min(1, V_exp / V_imp) 缩放所有进口需求，必要时可叠加部门供给上限。
- 政策：
    · 出口管制：将“基础出口”向量按部门系数乘子调整（0..1），影响对方进口的价值上限；
    · 进口关税/乘数：通过 import_multiplier 改变进口对偶价格 P_j^O，从而影响份额与需求。

注：此为可解释的、收敛性友好的“准动态”调整器，不代表显式跨期最优化。实际研究可替换为更严谨的动态结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import copy
import numpy as np
import torch

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
    DEFAULT_DEVICE,
    TORCH_DTYPE,
)


# ---------------------
# 状态表示
# ---------------------


class CountryState:
    """单国状态快照（支持批处理维度）。

    字段：
    - X_dom[i,j]: i 部门使用 j 的本国产品（中间品）
    - X_imp[i,k]: i 部门使用“可贸易集合第 k 个部门”的进口量（压缩列）
    - C_dom[j], C_imp[k]: 终端消费的国内/进口部分
    - price[j], imp_price[j]: 国内价格与进口对偶价格（含关税/运输）
    - export_base[j], export_actual[j]: 基础出口与当期实际成交出口量
    - output[j], income: 产出与收入
    """
    def __init__(self, X_dom, X_imp, C_dom, C_imp, price, imp_price, export_base, export_actual, output, income):
        self.X_dom = torch.as_tensor(X_dom, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.X_imp = torch.as_tensor(X_imp, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.C_dom = torch.as_tensor(C_dom, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.C_imp = torch.as_tensor(C_imp, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.price = torch.as_tensor(price, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.imp_price = torch.as_tensor(imp_price, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.export_base = torch.as_tensor(export_base, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.export_actual = torch.as_tensor(export_actual, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.output = torch.as_tensor(output, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE).clone()
        self.income = torch.as_tensor(income, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
        self.is_batch = self.X_dom.dim() == 3

    def ensure_batch(self, batch_size: int) -> "CountryState":
        if self.is_batch and self.X_dom.shape[0] == batch_size:
            return self
        def _repeat(t):
            if t.dim() == 3:
                return t
            return t.unsqueeze(0).repeat(batch_size, *([1] * (t.dim())))
        new_state = CountryState(
            _repeat(self.X_dom),
            _repeat(self.X_imp),
            _repeat(self.C_dom),
            _repeat(self.C_imp),
            _repeat(self.price),
            _repeat(self.imp_price),
            _repeat(self.export_base),
            _repeat(self.export_actual),
            _repeat(self.output),
            _repeat(self.income),
        )
        return new_state
    def detach(self) -> "CountryState":
        """返回一个新的 CountryState，其中所有 tensor 都已 detach 并转为 CPU (可选)，
        这里仅做 detach + clone 以切断计算图，避免 deepcopy 时的底层冲突。
        """
        return CountryState(
            self.X_dom.detach().clone(),
            self.X_imp.detach().clone(),
            self.C_dom.detach().clone(),
            self.C_imp.detach().clone(),
            self.price.detach().clone(),
            self.imp_price.detach().clone(),
            self.export_base.detach().clone(),
            self.export_actual.detach().clone(),
            self.output.detach().clone(),
            self.income.detach().clone(),
        )



def _merge_consumption(block: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    c_dom = torch.as_tensor(block["C_j"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    c_dom = c_dom + torch.as_tensor(block["C_I_j"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    c_imp = torch.as_tensor(block["C_O_j"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    return c_dom, c_imp


def _extract_export_base(partner_block: Dict[str, Dict[str, np.ndarray]]) -> torch.Tensor:
    interm = torch.as_tensor(partner_block["intermediate_inputs"]["X_O_ij"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    consumption = torch.as_tensor(partner_block["final_consumption"]["C_O_j"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    return torch.sum(interm, dim=0) + consumption


def _build_state(block: Dict[str, Dict[str, np.ndarray]], params: CountryParams,
                 export_base: torch.Tensor, tradable_mask: np.ndarray) -> CountryState:
    X_dom = torch.as_tensor(block["intermediate_inputs"]["X_ij"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    X_imp = torch.as_tensor(block["intermediate_inputs"]["X_O_ij"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    C_dom, C_imp = _merge_consumption(block["final_consumption"])
    prices = torch.as_tensor(block["prices"]["P_j"], dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
    output = compute_output(params, X_dom, X_imp, tradable_mask)
    income = compute_income(params, prices, output)
    imp_price = torch.ones_like(prices)
    return CountryState(X_dom, X_imp, C_dom, C_imp, prices, imp_price,
                        export_base, export_base, output, income)


class CountrySimulator:
    """单国的计划—价格—分配—再计算一周期更新器。

    关键接口：
    - _plan_demands：在给定价格下，以 Armington 与 Cobb–Douglas 关系计算计划需求（中间/消费）
    - _update_prices：基于供需缺口更新价格
    - _allocate_goods：供给不及时按比例分配到各用途
    - _allocate_imports_fx：按外汇约束缩放进口需求，并可施加部门供给上限
    - step：综合调用上一系列步骤，得到下一状态
    """
    def __init__(self, params: CountryParams, tradable_mask: np.ndarray, theta_price: float = 0.05):
        self.params = params
        self.tradable_mask = tradable_mask
        self.tradable_mask_tensor = torch.as_tensor(tradable_mask, dtype=torch.bool, device=DEFAULT_DEVICE)
        self.tau = torch.full((params.alpha.shape[0],), float(theta_price), dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)

    @staticmethod
    def _theta_from_usage(gamma: torch.Tensor, rho: torch.Tensor,
                          X_dom: torch.Tensor, X_imp: torch.Tensor) -> torch.Tensor:
        g = torch.clamp(gamma, 1e-6, 1 - 1e-6)
        r = rho
        clamped_dom = torch.clamp(X_dom, min=EPS)
        clamped_imp = torch.clamp(X_imp, min=EPS)
        dom = g * (clamped_dom ** r)
        imp = (1.0 - g) * (clamped_imp ** r)
        denom = torch.clamp(dom + imp, min=EPS)
        theta = dom / denom
        theta = torch.where(torch.abs(r) < 1e-10, g, theta)
        return torch.clamp(theta, 1e-6, 1 - 1e-6)

    @staticmethod
    def _theta_consumption(gamma_c: torch.Tensor, rho_c: torch.Tensor,
                           C_dom: torch.Tensor, C_imp: torch.Tensor) -> torch.Tensor:
        g = torch.clamp(gamma_c, 1e-6, 1 - 1e-6)
        r = rho_c
        c_dom = torch.clamp(C_dom, min=EPS)
        c_imp = torch.clamp(C_imp, min=EPS)
        dom = g * (c_dom ** r)
        imp = (1.0 - g) * (c_imp ** r)
        denom = torch.clamp(dom + imp, min=EPS)
        theta = dom / denom
        theta = torch.where(torch.abs(r) < 1e-10, g, theta)
        return torch.clamp(theta, 1e-6, 1 - 1e-6)

    def _plan_demands(self, state: CountryState, import_prices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        p = self.params
        outputs = compute_output(p, state.X_dom, state.X_imp, self.tradable_mask)
        lambdas = compute_marginal_cost(p, state.price, import_prices, self.tradable_mask)
        theta_prod_full = CountrySimulator._theta_from_usage(p.gamma, p.rho, state.X_dom, state.X_imp)
        theta_cons_full = CountrySimulator._theta_consumption(p.gamma_cons, p.rho_cons, state.C_dom, state.C_imp)
        theta_prod = torch.where(self.tradable_mask_tensor.unsqueeze(0), theta_prod_full, torch.ones_like(theta_prod_full))
        theta_cons = torch.where(self.tradable_mask_tensor, theta_cons_full, torch.ones_like(theta_cons_full))

        mask_a = p.alpha > 0
        log_alpha = torch.where(mask_a, torch.log(p.alpha), torch.zeros_like(p.alpha))
        log_lambda = safe_log(lambdas)[..., :, None]  # (..., n, 1)
        log_Y = safe_log(outputs)[..., :, None]       # (..., n, 1)
        log_price = safe_log(state.price)
        log_import_price = safe_log(import_prices)
        if log_price.dim() == 1:
            log_price = log_price.unsqueeze(0)  # (1,n)
            log_import_price = log_import_price.unsqueeze(0)
        elif log_price.dim() == 2:
            log_price = log_price.unsqueeze(-1)  # (batch,n,1)
            log_import_price = log_import_price.unsqueeze(-1)
        log_X_dom = safe_log(state.X_dom)
        log_X_imp = safe_log(state.X_imp)

        log_theta_prod = safe_log(theta_prod)
        log_one_minus_theta_prod = safe_log(1.0 - theta_prod)

        delta_dom = torch.zeros_like(state.X_dom)
        delta_imp = torch.zeros_like(state.X_imp)

        # tradable j
        trad_mask = self.tradable_mask_tensor.unsqueeze(0)
        delta_dom_trad = (log_alpha + log_lambda + log_theta_prod + log_Y - log_price - log_X_dom)
        delta_imp_trad = (log_alpha + log_lambda + log_one_minus_theta_prod + log_Y - log_import_price - log_X_imp)
        delta_dom = torch.where(mask_a & trad_mask, delta_dom_trad, delta_dom)
        delta_imp = torch.where(mask_a & trad_mask, delta_imp_trad, delta_imp)

        # non-tradable j
        non_trad_mask = (~self.tradable_mask_tensor).unsqueeze(0)
        delta_dom_non = (log_alpha + log_lambda + log_Y - log_price - log_X_dom)
        delta_dom = torch.where(mask_a & non_trad_mask, delta_dom_non, delta_dom)

        # 消费
        mask_b = p.beta > 0
        log_b = torch.where(mask_b, torch.log(p.beta), torch.zeros_like(p.beta))
        log_income = safe_log(state.income)
        log_price_vec = safe_log(state.price)
        log_import_price_vec = safe_log(import_prices)
        log_C_dom = safe_log(state.C_dom)
        log_C_imp = safe_log(state.C_imp)
        log_theta_cons = safe_log(theta_cons)
        log_one_minus_theta_cons = safe_log(1.0 - theta_cons)

        delta_c_dom = torch.zeros_like(state.C_dom)
        delta_c_imp = torch.zeros_like(state.C_imp)

        log_income_expand = log_income if log_income.dim() > 0 else log_income
        delta_c_dom_trad = log_b + log_income_expand + log_theta_cons - log_price_vec - log_C_dom
        delta_c_imp_trad = log_b + log_income_expand + log_one_minus_theta_cons - log_import_price_vec - log_C_imp
        delta_c_dom = torch.where(mask_b & self.tradable_mask_tensor, delta_c_dom_trad, delta_c_dom)
        delta_c_imp = torch.where(mask_b & self.tradable_mask_tensor, delta_c_imp_trad, delta_c_imp)

        delta_c_dom_non = log_b + log_income - log_price_vec - log_C_dom
        delta_c_dom = torch.where(mask_b & (~self.tradable_mask_tensor), delta_c_dom_non, delta_c_dom)

        planned_X_dom = state.X_dom * torch.exp(delta_dom)
        planned_X_imp = state.X_imp * torch.exp(delta_imp)
        planned_C_dom = state.C_dom * torch.exp(delta_c_dom)
        planned_C_imp = state.C_imp * torch.exp(delta_c_imp)
        return outputs, planned_X_dom, planned_X_imp, planned_C_dom, planned_C_imp, theta_prod, theta_cons

    def _update_prices(self, state: CountryState, outputs: torch.Tensor,
                       planned_X_dom: torch.Tensor, planned_C_dom: torch.Tensor) -> torch.Tensor:
        """价格更新：P' = P · exp(τ·(D - Y))，逐部门，D=中间+消费+基础出口。"""
        planned_total = planned_X_dom.sum(dim=0) + planned_C_dom + state.export_base
        demand_gap = planned_total - outputs
        delta_price = self.tau * demand_gap
        return state.price * torch.exp(delta_price)

    def _allocate_goods(self, outputs: torch.Tensor, planned_X_dom: torch.Tensor,
                        planned_C_dom: torch.Tensor, planned_export: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """国内供给分配：若总需求超出供给，按统一比例缩放。"""
        demand_total = torch.clamp(planned_X_dom.sum(dim=0) + planned_C_dom + planned_export, min=EPS)
        supply = torch.clamp(outputs, min=EPS)
        scale = torch.minimum(torch.tensor(1.0, dtype=TORCH_DTYPE, device=outputs.device), supply / demand_total)
        actual_X_dom = planned_X_dom * scale.unsqueeze(0)
        actual_C_dom = planned_C_dom * scale
        actual_export = planned_export * scale
        return actual_X_dom, actual_C_dom, actual_export

    def _allocate_imports_fx(self, import_prices: torch.Tensor, planned_X_imp: torch.Tensor,
                             planned_C_imp: torch.Tensor, export_value: torch.Tensor,
                             supply_cap: Optional[torch.Tensor] = None,
                             tradable_mask: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """进口外汇约束：以 s_fx=min(1, V_exp/V_imp) 缩放进口；可选供给上限逐部门裁剪。"""
        planned_value = torch.sum(planned_X_imp * import_prices.unsqueeze(0)) + torch.dot(planned_C_imp, import_prices)
        scale_fx = torch.tensor(1.0, dtype=TORCH_DTYPE, device=import_prices.device)
        if float(planned_value) > EPS:
            scale_fx = torch.minimum(export_value / planned_value, torch.tensor(1.0, dtype=TORCH_DTYPE, device=import_prices.device))
        sX = planned_X_imp * scale_fx
        sC = planned_C_imp * scale_fx
        if supply_cap is not None:
            mask = torch.as_tensor(tradable_mask if tradable_mask is not None else np.ones(sC.size(0), dtype=bool),
                                   device=import_prices.device)
            total = torch.sum(sX, dim=0) + sC
            cap = torch.clamp(supply_cap, min=0.0)
            scale_cap = torch.clamp(cap / torch.clamp(total, min=EPS), min=0.0, max=1.0)
            scale_cap = torch.where(mask, scale_cap, torch.tensor(1.0, dtype=TORCH_DTYPE, device=import_prices.device))
            sX = sX * scale_cap.unsqueeze(0)
            sC = sC * scale_cap
        return sX, sC

    def step(self, state: CountryState, import_prices: torch.Tensor,
             supply_cap: Optional[torch.Tensor] = None,
             tradable_mask: Optional[np.ndarray] = None) -> CountryState:
        """单期更新：计划→价格→国内分配→外贸约束→重算产出/收入。"""
        outputs, pX_dom, pX_imp, pC_dom, pC_imp, _, _ = self._plan_demands(state, import_prices)
        new_price = self._update_prices(state, outputs, pX_dom, pC_dom)
        aX_dom, aC_dom, aExport = self._allocate_goods(outputs, pX_dom, pC_dom, state.export_base)
        export_value = torch.dot(aExport, state.price)
        aX_imp, aC_imp = self._allocate_imports_fx(import_prices, pX_imp, pC_imp, export_value,
                                                   supply_cap=supply_cap, tradable_mask=tradable_mask)
        new_output = compute_output(self.params, aX_dom, aX_imp, self.tradable_mask)
        new_income = compute_income(self.params, new_price, new_output)
        return CountryState(aX_dom, aX_imp, aC_dom, aC_imp, new_price, import_prices,
                            state.export_base, aExport, new_output, new_income)


class TwoCountryDynamicSimulator:
    """两国动态仿真器（围绕静态均衡的政策冲击与响应路径）。

    初始化：
    - 从静态均衡提取“对方对我进口”形成本方基础出口向量（export_base）；
    - 设置进口对偶价格与关税乘数；
    - 设置政策事件日志。

    运行：每步对 H 和 F 同步执行 CountrySimulator.step（除非外部采用交替模式）。
    """
    def __init__(self, params: ModelParams, equilibrium: Dict[str, Dict], theta_price: float = 0.05, *, batch_size: int = 1):
        self.params = params
        tmask = np.zeros(params.home.alpha.shape[0], bool)
        tmask[params.tradable_idx] = True
        self.tradable_mask = tmask
        self.batch_size = max(int(batch_size), 1)

        export_H = _extract_export_base(equilibrium["F"])
        export_F = _extract_export_base(equilibrium["H"])

        self.home_state = _build_state(equilibrium["H"], params.home, export_H, tmask)
        self.foreign_state = _build_state(equilibrium["F"], params.foreign, export_F, tmask)
        if self.batch_size > 1:
            self.home_state = self.home_state.ensure_batch(self.batch_size)
            self.foreign_state = self.foreign_state.ensure_batch(self.batch_size)

        self.home_sim = CountrySimulator(params.home, tmask, theta_price)
        self.foreign_sim = CountrySimulator(params.foreign, tmask, theta_price)

        self.history: Dict[str, List[CountryState]] = {"H": [self.home_state], "F": [self.foreign_state]}

        self.baseline_export = {"H": self.home_state.export_base.clone(), "F": self.foreign_state.export_base.clone()}
        self.export_multiplier = {"H": torch.ones_like(self.baseline_export["H"]), "F": torch.ones_like(self.baseline_export["F"])}

        self.home_import_multiplier = params.home.import_cost.clone()
        self.foreign_import_multiplier = params.foreign.import_cost.clone()
        self.baseline_import_multiplier = {"H": self.home_import_multiplier.clone(), "F": self.foreign_import_multiplier.clone()}

        n = self.baseline_export["H"].size(0)
        mask_values = torch.as_tensor(tmask, dtype=TORCH_DTYPE, device=DEFAULT_DEVICE)
        k_H = mask_values.clone()
        k_F = mask_values.clone()
        self.import_cap_coeff: Dict[str, torch.Tensor] = {"H": k_H, "F": k_F}

        self.policy_events: List[Dict[str, Any]] = []

        pH, pF = self._import_prices()
        self.home_state.imp_price = pH.clone()
        self.foreign_state.imp_price = pF.clone()

    def _import_prices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.home_import_multiplier * self.foreign_state.price, self.foreign_import_multiplier * self.home_state.price

    def step(self) -> None:
        pH, pF = self._import_prices()
        cap_H = self.import_cap_coeff["H"] * self.foreign_state.export_actual
        cap_F = self.import_cap_coeff["F"] * self.home_state.export_actual
        new_H = self.home_sim.step(self.home_state, pH, supply_cap=cap_H, tradable_mask=self.tradable_mask)
        new_F = self.foreign_sim.step(self.foreign_state, pF, supply_cap=cap_F, tradable_mask=self.tradable_mask)
        self.home_state, self.foreign_state = new_H, new_F
        # 关键修复: 存入历史前先 detach，避免 deepcopy 时的 PyTorch/OpenMP 冲突
        self.history["H"].append(new_H.detach())
        self.history["F"].append(new_F.detach())

    def run(self, periods: int) -> None:
        for _ in range(periods):
            self.step()

    def summarize_history(self, batch_index: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
        def _pick(t: torch.Tensor) -> torch.Tensor:
            if t.dim() <= 1:
                return t
            return t[batch_index]

        def extract(states: List[CountryState]) -> Dict[str, np.ndarray]:
            price_mean = np.array([float(_pick(s.price).mean().detach().cpu()) for s in states], float)
            output_sum = np.array([float(_pick(s.output).sum().detach().cpu()) for s in states], float)
            income = np.array([float(_pick(s.income).detach().cpu()) for s in states], float)
            export_value = np.array([float((_pick(s.export_actual) * _pick(s.price)).sum().detach().cpu()) for s in states], float)
            import_value = np.array([float((_pick(s.C_imp).sum() + _pick(s.X_imp).sum()).detach().cpu()) for s in states], float)
            trade_balance = export_value - import_value
            import_value_val = np.array([float((_pick(s.imp_price) * (_pick(s.X_imp).sum(dim=0) + _pick(s.C_imp))).sum().detach().cpu()) for s in states], float)
            trade_balance_val = export_value - import_value_val
            income_growth = ((income / income[0]) - 1) * 100 if income[0] > 0 else np.zeros_like(income)
            output_growth = ((output_sum / output_sum[0]) - 1) * 100 if output_sum[0] > 0 else np.zeros_like(output_sum)
            return {
                "price_mean": price_mean,
                "output_sum": output_sum,
                "income": income,
                "export_value": export_value,
                "import_value": import_value,
                "trade_balance": trade_balance,
                "import_value_val": import_value_val,
                "trade_balance_val": trade_balance_val,
                "income_growth": income_growth,
                "output_growth": output_growth,
            }

        return {"H": extract(self.history["H"]), "F": extract(self.history["F"])}

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
        updated = torch.clamp(baseline * mult, min=0.0)
        target_state = self.home_state if country == "H" else self.foreign_state
        target_state.export_base = updated.clone()

    def apply_export_control(self, country: str, sector_factors: Dict[int, float], note: Optional[str] = None) -> None:
        """设置出口配额（基线乘子 0..1），记录事件。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        mult = self.export_multiplier[country]
        size = mult.size(0)
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
        """重置出口配额（全部或指定部门），记录事件。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        mult = self.export_multiplier[country]
        if sectors is None:
            mult[:] = 1.0
            touched = list(range(mult.size(0)))
        else:
            touched = []
            for sec in sectors:
                if not 0 <= sec < mult.size(0):
                    raise IndexError(f"部门索引 {sec} 超出范围 0-{mult.size(0) - 1}")
                mult[sec] = 1.0
                touched.append(int(sec))
        self._update_export_base(country)
        ev = {"type": "export_control_reset", "country": country, "sectors": touched}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def _import_multiplier_array(self, country: str) -> torch.Tensor:
        return self.home_import_multiplier if country == "H" else self.foreign_import_multiplier

    def apply_import_tariff(self, country: str, sector_rates: Dict[int, float], note: Optional[str] = None) -> None:
        """按百分比追加关税（累加到基线乘数），记录事件。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size(0)
        sanitized: Dict[int, float] = {}
        for sec, rate in sector_rates.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            target = baseline[sec] * (1.0 + float(rate))
            mult[sec] = torch.clamp(target, min=0.0)
            sanitized[int(sec)] = float(rate)
        ev = {"type": "import_tariff", "country": country, "sectors": sanitized}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def set_import_multiplier(self, country: str, sector_multipliers: Dict[int, float], *,
                               relative_to_baseline: bool = True, note: Optional[str] = None) -> None:
        """直接设置进口价格线性乘子（相对基线或绝对指定），记录事件。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size(0)
        sanitized: Dict[int, float] = {}
        for sec, m in sector_multipliers.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            target = baseline[sec] * float(m) if relative_to_baseline else torch.tensor(float(m), dtype=TORCH_DTYPE, device=mult.device)
            mult[sec] = torch.clamp(target, min=0.0)
            sanitized[int(sec)] = float(m)
        ev = {"type": "import_multiplier", "country": country, "relative": relative_to_baseline, "sectors": sanitized}
        if note:
            ev["note"] = note
        self._log_policy_event(ev)

    def reset_import_policies(self, country: str, sectors: Optional[List[int]] = None, note: Optional[str] = None) -> None:
        """重置进口相关政策（全部或指定部门），记录事件。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        baseline = self.baseline_import_multiplier[country]
        mult = self._import_multiplier_array(country)
        size = mult.size(0)
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
        """设置进口供给上限系数（用于极端冲击约束）；不可贸易部门强制为 0。"""
        country = country.upper()
        if country not in ("H", "F"):
            raise ValueError("country 需为 'H' 或 'F'")
        arr = self.import_cap_coeff[country]
        size = arr.size(0)
        for sec, val in sector_coeffs.items():
            if not 0 <= sec < size:
                raise IndexError(f"部门索引 {sec} 超出范围 0-{size - 1}")
            arr[sec] = float(max(val, 0.0)) if self.tradable_mask[sec] else 0.0

    def clone(self) -> "TwoCountryDynamicSimulator":
        new = object.__new__(TwoCountryDynamicSimulator)
        new.params = self.params
        new.tradable_mask = np.array(self.tradable_mask, copy=True)
        new.home_state = copy.deepcopy(self.home_state)
        new.foreign_state = copy.deepcopy(self.foreign_state)
        new.home_sim = copy.deepcopy(self.home_sim)
        new.foreign_sim = copy.deepcopy(self.foreign_sim)
        new.history = {
            "H": [copy.deepcopy(s) for s in self.history["H"]],
            "F": [copy.deepcopy(s) for s in self.history["F"]],
        }
        new.baseline_export = {"H": self.baseline_export["H"].clone(), "F": self.baseline_export["F"].clone()}
        new.export_multiplier = {"H": self.export_multiplier["H"].clone(), "F": self.export_multiplier["F"].clone()}
        new.home_import_multiplier = self.home_import_multiplier.clone()
        new.foreign_import_multiplier = self.foreign_import_multiplier.clone()
        new.baseline_import_multiplier = {"H": self.baseline_import_multiplier["H"].clone(), "F": self.baseline_import_multiplier["F"].clone()}
        new.import_cap_coeff = {"H": self.import_cap_coeff["H"].clone(), "F": self.import_cap_coeff["F"].clone()}
        new.policy_events = list(self.policy_events)
        return new

    def fork(self, keep_history: bool = False, batch_size: Optional[int] = None) -> "TwoCountryDynamicSimulator":
        """轻量复制：仅保留当前状态，历史清零；可选调整 batch_size。

        - keep_history=True 时退回 clone()（深拷贝历史）。
        - 默认仅复制当前帧，history 只保留当前状态，适用于搜索前瞻等无需过往轨迹的场景。
        """
        if keep_history:
            return self.clone()

        new = object.__new__(TwoCountryDynamicSimulator)
        new.params = self.params
        new.tradable_mask = np.array(self.tradable_mask, copy=True)
        new.batch_size = int(batch_size) if batch_size is not None else self.batch_size

        home = self.home_state.detach()
        foreign = self.foreign_state.detach()
        if new.batch_size > 1:
            home = home.ensure_batch(new.batch_size)
            foreign = foreign.ensure_batch(new.batch_size)
        new.home_state = home
        new.foreign_state = foreign

        # 轻量复制仿真器内部组件
        new.home_sim = copy.deepcopy(self.home_sim)
        new.foreign_sim = copy.deepcopy(self.foreign_sim)

        new.history = {"H": [home], "F": [foreign]}
        new.baseline_export = {"H": self.baseline_export["H"].clone(), "F": self.baseline_export["F"].clone()}
        new.export_multiplier = {"H": self.export_multiplier["H"].clone(), "F": self.export_multiplier["F"].clone()}
        new.home_import_multiplier = self.home_import_multiplier.clone()
        new.foreign_import_multiplier = self.foreign_import_multiplier.clone()
        new.baseline_import_multiplier = {
            "H": self.baseline_import_multiplier["H"].clone(),
            "F": self.baseline_import_multiplier["F"].clone(),
        }
        new.import_cap_coeff = {"H": self.import_cap_coeff["H"].clone(), "F": self.import_cap_coeff["F"].clone()}
        new.policy_events = list(self.policy_events)
        return new


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
    theta_price: float = 0.05
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


def bootstrap_simulator(params_raw: Dict[str, Dict], theta_price: float = 0.05) -> TwoCountryDynamicSimulator:
    """便捷启动：以默认精度求解静态均衡并构造仿真器。"""
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params, max_iterations=400, tolerance=1e-8)
    return TwoCountryDynamicSimulator(params, eqm, theta_price)


def simulate(config: SimulationConfig, params_raw: Optional[Dict[str, Dict]] = None) -> TwoCountryDynamicSimulator:
    """运行带时间线的两国仿真：按 config 生成/合并 PolicyEvent，逐期施加并推进。"""
    if params_raw is None:
        params_raw = create_symmetric_parameters()
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params, max_iterations=config.solver_max_iter, tolerance=config.solver_tol)
    sim = TwoCountryDynamicSimulator(params, eqm, theta_price=config.theta_price)

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
                elif ev.kind == "export_tariff":
                    actor = (ev.actor or "").upper()
                    target = "H" if actor == "F" else ("F" if actor == "H" else None)
                    if target is not None:
                        sim.reset_import_policies(target, sectors=list(ev.sectors.keys()), note=f"结束期 {t} 还原对方进口税(映射自出口税)")

        if t in start_bucket:
            for ev in start_bucket[t]:
                if ev.kind == "export_quota":
                    sim.apply_export_control(ev.actor, ev.sectors, note=ev.note)
                elif ev.kind == "import_tariff":
                    sim.apply_import_tariff(ev.actor, ev.sectors, note=ev.note)
                elif ev.kind == "import_multiplier":
                    sim.set_import_multiplier(ev.actor, ev.sectors, relative_to_baseline=True, note=ev.note)
                elif ev.kind == "export_tariff":
                    actor = (ev.actor or "").upper()
                    target = "H" if actor == "F" else ("F" if actor == "H" else None)
                    if target is not None:
                        sim.apply_import_tariff(target, ev.sectors, note=(ev.note or "") + " | mirror of export_tariff")
                        sim._log_policy_event({"type": "export_tariff", "country": actor, "sectors": dict(ev.sectors), "note": ev.note})

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
