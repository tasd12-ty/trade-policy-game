"""
博弈模块（Game Module）
========================
本模块实现两国最优回应博弈的核心逻辑：
- OptimalResponseAgent: 最优回应智能体，使用优化器搜索最优政策
- GameSimulator: 博弈仿真器，管理决策循环与状态记录

目标函数参考 analysis/latex/bo_response_math.tex 中的数学定义。
"""

import numpy as np
import copy
from typing import Dict, Any, List, Tuple
from analysis.model.sim import TwoCountryDynamicSimulator
# 优化器策略在 analysis/optimizers.py 中定义
from analysis.optimizers import (
    Optimizer,
    BayesianOptimizer,
    SPSAOptimizer,
    GradientDescentOptimizer,
    HAS_BAYES_OPT,
)

# 从数学文档中提取的默认常数
DEFAULT_LOOKAHEAD = 15     # 默认前瞻步数（评估窗口 H）
DEFAULT_DECISION_INTERVAL = 10  # 默认决策间隔
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)  # 目标函数权重：(收入, 贸易差额, 价格稳定)


class OptimalResponseAgent:
    """
    最优回应智能体。
    
    该智能体在每个决策点使用优化器搜索最优贸易政策（关税、出口配额），
    目标是最大化加权目标函数（收入增长 + 贸易差额 - 价格波动）。
    
    属性:
        name: 智能体名称 ('H' 或 'F')
        n_sectors: 部门数量
        optimizer: 优化器实例
        w_income, w_tb, w_price: 目标函数权重
        lookahead: 前瞻评估步数
        reciprocity_factor: 对等反制系数 (A)，控制关税下界
        max_tariff: 最大允许关税率
        min_multiplier: 最小出口乘子（0.5 表示最多限制一半出口）
    """
    
    def __init__(
        self,
        name: str,
        n_sectors: int,
        optimizer: Optimizer = None,
        weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
        lookahead_steps: int = DEFAULT_LOOKAHEAD,
        reciprocity_factor: float = 1.0,  # 对等反制系数 A（文档定义）
        max_tariff: float = 0.5,
        min_export_multiplier: float = 0.5  # m ∈ [0.5, 1.0]，0 会完全中断贸易
    ):
        self.name = name
        self.n_sectors = n_sectors
        # 默认优先使用 BO；若未安装 bayes_opt，则退化为 SPSA
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = BayesianOptimizer() if HAS_BAYES_OPT else SPSAOptimizer()
        self.w_income, self.w_tb, self.w_price = weights
        self.lookahead = lookahead_steps
        self.reciprocity_factor = reciprocity_factor
        self.max_tariff = max_tariff
        self.min_multiplier = min_export_multiplier
        
    def decide(
        self,
        sim: TwoCountryDynamicSimulator,
        opponent_prev_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        确定本智能体的最优行动（关税率、出口乘子）。
        
        参数:
            sim: 当前仿真器状态
            opponent_prev_action: 对手上一轮的行动（用于对等约束）
        
        返回:
            包含 'import_tariff' 和 'export_quota' 的行动字典
        """
        # === 1. 确定决策变量边界 ===
        # 对手行动结构: {'import_tariff': {部门: 税率}, ...}
        opp_tariffs = opponent_prev_action.get('import_tariff', {})
        
        bounds = []
        # 参数布局: [Tariff_S0, ..., Tariff_Sn, Multiplier_S0, ..., Multiplier_Sn]
        # 共 2 * n_sectors 个参数
        
        # 关税边界（考虑对等约束）
        for i in range(self.n_sectors):
            # 对等反制：若对手对部门 i 征税 tax1，则本方关税下界为 A * tax1
            opp_t = opp_tariffs.get(i, 0.0)
            lower_b = min(self.reciprocity_factor * opp_t, self.max_tariff)
            upper_b = self.max_tariff
            bounds.append((lower_b, upper_b))
            
        # 出口乘子边界
        # m = 1.0 表示自由贸易，m < 1.0 表示出口限制
        for i in range(self.n_sectors):
            bounds.append((self.min_multiplier, 1.0))
            
        bounds_arr = np.array(bounds)
        
        # === 2. 定义目标函数 ===
        def objective(params):
            """
            目标函数：评估给定政策参数的效果。
            
            通过 fork 仿真器、应用政策、运行前瞻仿真来计算目标值。
            """
            # 解码参数
            n = self.n_sectors
            tariffs = params[:n]
            multipliers = params[n:]
            
            # 构建行动字典（过滤零值以避免不必要的政策事件）
            action = {
                "import_tariff": {i: t for i, t in enumerate(tariffs) if t > 1e-4},
                "export_quota": {i: m for i, m in enumerate(multipliers) if abs(m - 1.0) > 1e-4}
            }
            
            # 克隆仿真器（优先使用 fork 以提高效率）
            fork = getattr(sim, "fork", None)
            cloner = getattr(sim, "clone", None)
            
            if callable(fork):
                sim_run = fork(keep_history=False)
            elif callable(cloner):
                sim_run = cloner()
            else:
                sim_run = copy.deepcopy(sim)
                
            # 应用行动
            sim_run.apply_action(self.name, action)
            
            # === 前瞻仿真 ===
            # 获取初始状态用于归一化
            h0 = sim_run.home_state
            f0 = sim_run.foreign_state
            
            this_state_0 = h0 if self.name == 'H' else f0
            opp_state_0 = f0 if self.name == 'H' else h0
            
            I0 = this_state_0.income.item()  # 初始收入
            P0 = this_state_0.price.clone()  # 初始价格向量
            S_scale = I0  # 贸易差额归一化系数（使用初始收入作为标度）
            
            incomes = []
            trade_balances = []
            prices = []  # 价格指数序列
            
            for _ in range(self.lookahead):
                sim_run.step()
                
                curr_h = sim_run.home_state
                curr_f = sim_run.foreign_state
                curr_me = curr_h if self.name == 'H' else curr_f
                
                # 记录收入
                incomes.append(curr_me.income.item())
                
                # 计算贸易差额
                # TB = 出口价值 - 进口价值
                exp_val = (curr_me.export_actual * curr_me.price).sum()
                imp_qty = curr_me.X_imp.sum(dim=0) + curr_me.C_imp
                imp_val = (curr_me.imp_price * imp_qty).sum()
                tb = exp_val - imp_val
                trade_balances.append(tb.item())
                
                # 记录价格指数 (P_t / P_0 的均值)
                pi_t = (curr_me.price / P0).mean().item()
                prices.append(pi_t)
                
            # === 计算指标 ===
            # 1. 收入增长率: (1/H) * Σ(I_t/I_0 - 1)
            i_growth = np.mean([i/I0 - 1 for i in incomes])
            
            # 2. 归一化贸易差额: (1/H) * Σ(TB_t/S_scale)
            tb_norm = np.mean([tb/S_scale for tb in trade_balances])
            
            # 3. 价格稳定性: -std(价格指数)，负号使波动越小越好
            p_std = np.std(prices)
            
            # 总目标值: J = w1 * 收入增长 + w2 * 贸易差额 - w3 * 价格波动
            score = self.w_income * i_growth + self.w_tb * tb_norm - self.w_price * p_std
            
            return score

        # === 3. 执行优化 ===
        # 初始猜测：中性政策（零关税、完全自由出口）
        n = self.n_sectors
        init = np.array([0.0]*n + [1.0]*n)
        
        best_params, best_score, _ = self.optimizer.optimize(
            objective_function=objective,
            bounds=bounds_arr,
            initial_guess=init,
            n_iter=10  # 迭代次数，可根据需要调整
        )
        
        # === 4. 转换为行动格式 ===
        best_tariffs = best_params[:n]
        best_mults = best_params[n:]
        
        final_action = {
            "import_tariff": {i: t for i, t in enumerate(best_tariffs)},
            "export_quota": {i: m for i, m in enumerate(best_mults)}
        }
        
        return final_action


class GameSimulator:
    """
    博弈仿真器。
    
    管理两国智能体的交互循环：
    1. 每隔固定步数（decision_interval），双方同时决策
    2. 决策基于对手上一轮的行动（囚徒困境式信息结构）
    3. 记录决策日志和经济指标
    
    属性:
        sim: 底层经济仿真器
        agent_h, agent_f: 两国智能体
        logs: 日志记录列表
        last_action_h, last_action_f: 上一轮行动（用于对等响应）
    """
    
    def __init__(
        self, 
        sim: TwoCountryDynamicSimulator, 
        agent_h: OptimalResponseAgent, 
        agent_f: OptimalResponseAgent
    ):
        self.sim = sim
        self.agent_h = agent_h
        self.agent_f = agent_f
        self.logs = []
        
        # 行动历史跟踪
        self.last_action_h = {}
        self.last_action_f = {}
        
    def run(self, steps: int = 100, decision_interval: int = 20):
        """
        运行博弈仿真。
        
        参数:
            steps: 总仿真步数（假设已预热）
            decision_interval: 决策间隔（每多少步决策一次）
        """
        for t in range(1, steps + 1):
            is_decision_step = (t % decision_interval == 0)
            
            if is_decision_step:
                print(f"步骤 {t}: 智能体正在优化决策...")
                
                # 同时决策：各方基于对方的上一轮行动做出响应
                action_h = self.agent_h.decide(self.sim, self.last_action_f)
                action_f = self.agent_f.decide(self.sim, self.last_action_h)
                
                # 应用行动（政策持续生效直到下次变更）
                self.sim.apply_action("H", action_h)
                self.sim.apply_action("F", action_f)
                
                # 更新历史
                self.last_action_h = action_h
                self.last_action_f = action_f
                
                # 记录决策
                self._log_decision(t, "H", action_h)
                self._log_decision(t, "F", action_f)
            
            # 推进仿真一步
            self.sim.step()
            self._log_metrics(t)
            
    def _log_decision(self, t: int, actor: str, action: Dict[str, Any]):
        """记录决策日志。"""
        tariffs = list(action.get('import_tariff', {}).values())
        avg_t = np.mean(tariffs) if tariffs else 0
        self.logs.append({
            "t": t,
            "type": "decision",
            "actor": actor,
            "avg_tariff": avg_t
        })
        
    def _log_metrics(self, t: int):
        """记录经济指标日志。"""
        h = self.sim.home_state
        f = self.sim.foreign_state
        self.logs.append({
            "t": t,
            "type": "metric",
            "h_income": h.income.item(),
            "f_income": f.income.item()
        })


__all__ = [
    "OptimalResponseAgent",
    "GameSimulator",
    "DEFAULT_LOOKAHEAD",
    "DEFAULT_DECISION_INTERVAL",
    "DEFAULT_WEIGHTS",
]
