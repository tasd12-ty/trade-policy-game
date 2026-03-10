"""基于梯度的策略优化智能体（torch 实现）。

参考 grad_op/analysis/optimization/grad_game.py 的
_optimize_static_best_response() 和 DifferentiableObjective。

核心思路：
1. 从当前 NumPy 状态快照出发
2. 用 torch 实现可微分的简化动态前瞻
3. Adam 优化关税/配额参数
4. multi_start 多起点选最优

依赖：torch, numpy, agent_interface.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent_interface import PolicyAgent

EPS_T = 1e-9


# ==================== Torch 可微分工具 ====================

def _armington_share_torch(
    gamma: torch.Tensor,
    p_dom: torch.Tensor,
    p_for: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """CES Armington 份额（可微分）。

    θ = γ^σ · p_d^{1-σ} / (γ^σ · p_d^{1-σ} + (1-γ)^σ · p_f^{1-σ})
    σ = 1/(1-ρ)
    """
    sigma = 1.0 / (1.0 - rho + EPS_T)
    w_d = (gamma ** sigma) * (p_dom ** (1.0 - sigma))
    w_f = ((1.0 - gamma) ** sigma) * (p_for ** (1.0 - sigma))
    share = w_d / (w_d + w_f + EPS_T)
    return share.clamp(1e-8, 1.0 - 1e-8)


def _armington_price_torch(
    gamma: torch.Tensor,
    p_dom: torch.Tensor,
    p_for: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """CES 对偶价格（可微分）。

    P* = [γ^σ · p_d^{1-σ} + (1-γ)^σ · p_f^{1-σ}]^{1/(1-σ)}
    """
    sigma = 1.0 / (1.0 - rho + EPS_T)
    inner = (gamma ** sigma) * (p_dom ** (1.0 - sigma)) + \
            ((1.0 - gamma) ** sigma) * (p_for ** (1.0 - sigma))
    return (inner + EPS_T) ** (1.0 / (1.0 - sigma + EPS_T))


# ==================== 可微分前瞻模型 ====================

class DifferentiableForward:
    """可微分的简化动态前瞻。

    从 NumPy 状态快照出发，在 torch 中执行 lookahead 步。
    关税参数为 requires_grad=True 的 torch tensor。

    简化：仅模拟价格调整 + Armington 反应 + 产出 + 收入。
    不含完整的分配逻辑（eq 20-24），但足以计算目标函数梯度。
    """

    def __init__(
        self,
        # 国家参数（NumPy）
        alpha: np.ndarray,    # (Nl, Nl+M)
        gamma: np.ndarray,    # (Nl, Nl)
        rho: np.ndarray,      # (Nl, Nl)
        beta: np.ndarray,     # (Nl,)
        A: np.ndarray,        # (Nl,)
        gamma_cons: np.ndarray,  # (Nl,)
        rho_cons: np.ndarray,    # (Nl,)
        L: np.ndarray,        # (M,)
        Ml: int,
        M_factors: int,
        # 动态参数
        tau: float = 0.1,
        normalize_gap: bool = True,
    ):
        self.Nl = alpha.shape[0]
        self.Ml = Ml
        self.M = M_factors
        self.tau = tau
        self.normalize_gap = normalize_gap

        # 转换为 torch（不需要梯度的参数）
        self.alpha = torch.tensor(alpha, dtype=torch.float64)
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.rho = torch.tensor(rho, dtype=torch.float64)
        self.beta = torch.tensor(beta, dtype=torch.float64)
        self.A = torch.tensor(A, dtype=torch.float64)
        self.gamma_cons = torch.tensor(gamma_cons, dtype=torch.float64)
        self.rho_cons = torch.tensor(rho_cons, dtype=torch.float64)
        self.L = torch.tensor(L, dtype=torch.float64)

    def forward(
        self,
        price_init: np.ndarray,       # (Nl,) 当前国内价格
        output_init: np.ndarray,       # (Nl,) 当前产出
        income_init: float,            # 当前收入
        export_base: np.ndarray,       # (Nl,) 出口基线
        partner_price: np.ndarray,     # (Nl,) 对手国内价格
        base_import_cost: np.ndarray,  # (Nl,) 基线进口成本乘子
        tariff_param: torch.Tensor,    # (n_active,) 关税率参数
        active_sectors: List[int],     # 可调整部门索引
        lookahead: int,
    ) -> Dict[str, torch.Tensor]:
        """可微分前瞻：返回 income/tb/price 序列的 torch tensor。

        简化策略（保证数值稳定性）：
        - 产出 output 保持固定（不重新计算），避免循环反馈爆炸
        - 主要通道：tariff → imp_price → Armington份额 → 国内需求 → 价格调整 → 收入
        - 与真实引擎 dynamics.py 一致的 tanh 阻尼价格更新
        """
        Nl = self.Nl

        # 初始化 torch 状态
        price = torch.tensor(price_init[:Nl], dtype=torch.float64)
        output = torch.tensor(output_init[:Nl], dtype=torch.float64)  # 固定不变
        income = torch.tensor(income_init, dtype=torch.float64)
        exp_base = torch.tensor(export_base[:Nl], dtype=torch.float64)
        p_partner = torch.tensor(partner_price[:Nl], dtype=torch.float64)
        base_ic = torch.tensor(base_import_cost[:Nl], dtype=torch.float64)

        alpha_prod = self.alpha[:, :Nl]  # (Nl, Nl)

        # 构造进口价格：base_cost * (1 + tariff) * partner_price
        import_mult = torch.ones(Nl, dtype=torch.float64)
        for idx, s in enumerate(active_sectors):
            if s < Nl:
                import_mult[s] = 1.0 + tariff_param[idx]
        imp_price = base_ic * import_mult * p_partner

        # 收集轨迹
        incomes = []
        tb_vals = []
        price_indices = []

        for _t in range(lookahead):
            # -- Armington 国内份额 --
            # 生产用中间投入份额
            theta_prod = _armington_share_torch(
                self.gamma[0, :], price, imp_price, self.rho[0, :],
            )
            # 非贸易品：θ=1（纯国内）
            theta_prod = theta_prod.clone()
            theta_prod[:self.Ml] = 1.0

            # 消费用份额
            theta_cons = _armington_share_torch(
                self.gamma_cons, price, imp_price, self.rho_cons,
            )
            theta_cons = theta_cons.clone()
            theta_cons[:self.Ml] = 1.0

            # -- 国内需求 --
            # 中间需求：Σ_i α_{ij} · Y_i · θ_j
            int_demand = (alpha_prod * output.unsqueeze(1)).sum(0) * theta_prod

            # 消费需求：β_j · θ_{cj} · I / P_j
            cons_demand = self.beta * theta_cons * income / (price + EPS_T)

            # 总国内需求 = 中间 + 消费 + 出口
            total_demand = int_demand + cons_demand + exp_base

            # -- 供需缺口 → 价格更新 (eq 19) --
            gap = total_demand - output
            if self.normalize_gap:
                gap = gap / (output + EPS_T)

            cap = 3.0
            delta = cap * torch.tanh(self.tau * gap / cap)
            price = price * torch.exp(delta)
            price = price.clamp(0.01, 100.0)

            # -- 收入 = 要素报酬 ≈ Σ_k α_{Nl+k} · P · Y --
            income_new = torch.zeros(1, dtype=torch.float64)
            for k in range(self.M):
                income_new = income_new + (self.alpha[:, Nl + k] * price * output).sum()
            income = income_new.squeeze().clamp(min=EPS_T)

            # -- 贸易差额 --
            export_val = (price * exp_base).sum()
            # 进口值 = 中间进口 + 消费进口
            import_share = 1.0 - theta_prod
            int_import_val = (
                (alpha_prod * output.unsqueeze(1)).sum(0) * import_share * imp_price
            ).sum()
            cons_imp_val = (self.beta * (1.0 - theta_cons) * income).sum()
            tb = export_val - int_import_val - cons_imp_val

            incomes.append(income)
            tb_vals.append(tb)
            price_indices.append(price.mean())

        return {
            "income": torch.stack(incomes),
            "trade_balance": torch.stack(tb_vals),
            "price_index": torch.stack(price_indices),
            "income_0": torch.tensor(income_init, dtype=torch.float64),
        }


def _compute_objective(
    trajectory: Dict[str, torch.Tensor],
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    """计算加权目标函数（与 grad_op DifferentiableObjective 一致）。

    J = w_inc * mean(income_growth) + w_tb * mean(tb/income_0) + w_stab * (-std(price_idx))
    """
    w_inc, w_tb, w_stab = weights
    income_0 = trajectory["income_0"]
    incomes = trajectory["income"]
    tb = trajectory["trade_balance"]
    price_idx = trajectory["price_index"]

    # 收入增长
    growth = (incomes / (income_0 + EPS_T)) - 1.0
    score_income = growth.mean()

    # 贸易差额归一化
    score_tb = (tb / (income_0 + 1.0)).mean()

    # 价格稳定性
    if len(price_idx) > 1:
        p_norm = price_idx / (price_idx[0] + EPS_T)
        score_stab = -p_norm.std()
    else:
        score_stab = torch.tensor(0.0, dtype=torch.float64)

    return w_inc * score_income + w_tb * score_tb + w_stab * score_stab


# ==================== 梯度策略智能体 ====================

class GradientPolicyAgent(PolicyAgent):
    """基于梯度的策略优化智能体（Adam + 可微分前瞻）。

    参考 grad_op 的 _optimize_static_best_response()。

    每次 decide() 时：
    1. 从当前仿真状态快照
    2. 创建 K 个关税参数起点（multi_start）
    3. 对每个起点用 Adam 优化 iterations 步
    4. 选择目标函数最大的结果

    参数：
        sandbox:           EconomicSandbox 引用（获取当前状态）
        country:           本国 "H" 或 "F"
        active_sectors:    可调整的部门索引
        lookahead_periods: 前瞻步数
        lr:                Adam 学习率
        iterations:        每个起点的优化步数
        multi_start:       起点数量
        start_strategy:    "current", "noisy_current", "random"
        max_tariff:        关税率上限
        min_quota:         配额下限（暂未使用）
        objective_weights: (income, trade_balance, stability) 权重
    """

    def __init__(
        self,
        sandbox,
        country: str,
        active_sectors: List[int],
        lookahead_periods: int = 12,
        lr: float = 0.01,
        iterations: int = 200,
        multi_start: int = 8,
        start_strategy: str = "noisy_current",
        max_tariff: float = 1.0,
        min_quota: float = 0.0,
        objective_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.sandbox = sandbox
        self.country = country.upper()
        self.active_sectors = active_sectors
        self.lookahead = lookahead_periods
        self.lr = lr
        self.iterations = iterations
        self.multi_start = multi_start
        self.start_strategy = start_strategy
        self.max_tariff = max_tariff
        self.min_quota = min_quota
        self.weights = objective_weights

        self._last_obs: Optional[Dict] = None
        self._current_tariff: Dict[int, float] = {}
        self._round = 0

    def observe(self, context: Dict) -> None:
        self._last_obs = context
        self._round += 1

    def decide(self) -> Dict[str, Any]:
        """执行梯度优化搜索最优关税率。"""
        if self._last_obs is None:
            return {"tariff": {}, "quota": {}}

        sim = self.sandbox.sim
        c = self.country
        opp = "F" if c == "H" else "H"

        # 获取当前状态
        state = sim.history[c][-1]
        opp_state = sim.history[opp][-1]
        cp = sim.params.home if c == "H" else sim.params.foreign
        opp_cp = sim.params.foreign if c == "H" else sim.params.home
        Nl = cp.Nl

        # 构建可微分前瞻
        fwd = DifferentiableForward(
            alpha=np.asarray(cp.alpha, dtype=float),
            gamma=np.asarray(cp.gamma, dtype=float),
            rho=np.asarray(cp.rho, dtype=float),
            beta=np.asarray(cp.beta, dtype=float),
            A=np.asarray(cp.A, dtype=float),
            gamma_cons=np.asarray(cp.gamma_cons, dtype=float),
            rho_cons=np.asarray(cp.rho_cons, dtype=float),
            L=np.asarray(cp.L, dtype=float),
            Ml=int(cp.Ml),
            M_factors=int(cp.M_factors),
            tau=float(sim._tau) if np.isscalar(sim._tau) else float(sim._tau[0]),
            normalize_gap=bool(sim._normalize_gap),
        )

        n_active = len(self.active_sectors)

        # 当前关税率
        current_rates = np.array([
            self._current_tariff.get(s, 0.0) for s in self.active_sectors
        ], dtype=float)

        # multi_start 优化
        best_tariff = None
        best_obj = -float("inf")

        for k in range(self.multi_start):
            # 初始化
            if self.start_strategy == "current":
                init = current_rates.copy()
            elif self.start_strategy == "noisy_current":
                init = current_rates + np.random.randn(n_active) * 0.1
            elif self.start_strategy == "random":
                init = np.random.uniform(0, self.max_tariff, n_active)
            else:
                init = current_rates.copy()

            init = np.clip(init, 0.0, self.max_tariff)

            tariff_param = torch.tensor(init, dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([tariff_param], lr=self.lr)

            for _i in range(self.iterations):
                optimizer.zero_grad()

                traj = fwd.forward(
                    price_init=state.price[:Nl],
                    output_init=state.output[:Nl],
                    income_init=state.income,
                    export_base=state.export_base[:Nl],
                    partner_price=opp_state.price[:opp_cp.Nl],
                    base_import_cost=np.asarray(cp.import_cost, dtype=float),
                    tariff_param=tariff_param,
                    active_sectors=self.active_sectors,
                    lookahead=self.lookahead,
                )

                obj = _compute_objective(traj, self.weights)
                loss = -obj  # 最大化 → 最小化负值
                loss.backward()
                optimizer.step()

                # 投影到约束域
                with torch.no_grad():
                    tariff_param.data.clamp_(0.0, self.max_tariff)

            final_obj = obj.item()
            if final_obj > best_obj:
                best_obj = final_obj
                best_tariff = tariff_param.detach().numpy().copy()

        # 构造决策
        tariff_dict = {}
        for idx, s in enumerate(self.active_sectors):
            rate = float(best_tariff[idx])
            if rate > 0.005:  # 忽略微小关税
                tariff_dict[s] = round(rate, 4)

        self._current_tariff = tariff_dict

        print(f"  [{self.country}] Round {self._round}: "
              f"tariff={tariff_dict}, obj={best_obj:.6f}")

        return {"tariff": tariff_dict, "quota": {}}
