"""交互式博弈模拟框架

实现两国博弈的交互式优化框架，支持：
- 同时决策：两国在每轮博弈中同时优化各自策略
- 多种目标类型：标准（自身福利）vs. 相对（零和优势）
- 策略模式：独立优化 vs. 对等反制约束
- 基于 SPSA 的黑箱优化

核心流程：
1. 双方基于当前仿真状态和 Lookahead 优化策略
2. 同时应用策略
3. 推进仿真若干期
4. 重复上述过程

使用方法：
    config = GameConfig(objective_type_H="standard", strategy_mode_F="reciprocal")
    game = InteractiveGame(config)
    for _ in range(rounds):
        game.step()
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from time import perf_counter
from analysis.model import TwoCountryDynamicSimulator, bootstrap_simulator, create_symmetric_parameters
from analysis.optimization.objective import (
    evaluate_snapshot_objective,
    decision_vector_for_sectors,
    apply_per_sector_from_x,
    Country,
)
from analysis.optimization.spsa_opt import spsa, SPSAConfig

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    objective_type_H: Literal["standard", "relative"] = "standard"
    objective_type_F: Literal["standard", "relative"] = "standard"
    
    # 策略模式: 'independent' (独立优化), 'reciprocal' (对等反制 constraint)
    strategy_mode_H: Literal["independent", "reciprocal"] = "independent"
    strategy_mode_F: Literal["independent", "reciprocal"] = "independent"
    
    # 反制系数 (alpha): 下限 = alpha * Opponent_Last_Tariff
    reciprocal_alpha: float = 1.0
    
    # 优化参数
    opt_horizon: int = 20  # 前瞻评估步数（Lookahead）
    opt_iter: int = 400     # SPSA 迭代步数（演示默认较小以加速）
    opt_perturb: float = 0.05
    opt_lr: float = 0.01

    # SPSA 超参（参考 Spall 经典取值：alpha≈0.602、gamma≈0.101）
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_seed: int = 42
    # 多次随机重启：每次用不同 seed 跑一遍 SPSA，取最优解（会增加耗时）
    spsa_restarts: int = 1
    # 多起点（multi-start）：从多个不同初值出发分别跑 SPSA，取最优解（会显著增加耗时）
    # 说明：restarts 改变随机方向序列；multi_start 改变初始点，两者可叠加。
    spsa_multi_start: int = 1
    # 起点策略：
    # - "neutral"：使用 x0（中性政策）
    # - "random"：在边界内均匀随机采样
    # - "noisy_neutral"：在 x0 附近加噪声（按区间尺度），再裁剪到边界
    spsa_start_strategy: Literal["neutral", "random", "noisy_neutral"] = "neutral"
    # noisy_neutral 的相对噪声强度（按变量区间宽度缩放），例如 0.05 表示 5% 量级扰动
    spsa_start_noise: float = 0.05

    # 决策变量维度控制
    # 默认只优化前若干个可贸易部门（演示用）；设为 None 表示优化全部可贸易部门（会明显变慢）
    max_sectors: Optional[int] = 2
    # 关税上限（例如 2.0 表示 +200%）
    tau_max: float = 2.0
    # 价格项放大倍数：过大时会让策略过于“保守”（倾向于不动政策以维稳）
    price_scale: float = 10_000.0
    
    # 仿真步长 (决策间隔)
    step_interval: int = 10
    
    # 权重顺序：(收入, 贸易差额, 价格稳定性)
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class InteractiveGame:
    def __init__(self, config: GameConfig, initial_sim: Optional[TwoCountryDynamicSimulator] = None):
        self.config = config
        if initial_sim:
            self.sim = initial_sim
        else:
            params = create_symmetric_parameters()
            self.sim = bootstrap_simulator(params)
            # Burn-in a bit?
            self.sim.run(10)
            
        self.history_policies: Dict[Country, List[Dict]] = {"H": [], "F": []}
        # 记录上一轮实施的决策向量 (x)，便于反制计算
        self.last_x: Dict[Country, Optional[np.ndarray]] = {"H": None, "F": None}
        self.round_idx = 0
        # 时间统计：每轮 step 的耗时分解（秒）
        # 字段示例：round_idx, period, optimize_h_s, optimize_f_s, apply_s, simulate_s, total_s
        self.timing_records: List[Dict[str, float]] = []

    def step(self):
        """执行一轮博弈：H/F 决策 -> 仿真 -> 记录。

        同时记录时间统计信息，便于评估不同超参/场景的计算代价。
        """
        logger.info(f"--- Round {self.round_idx} Start ---")
        t_start = perf_counter()
        period = getattr(self.sim, "_current_period", lambda: None)()
        
        # 1. 双方同时决策 (Nash-like or simultaneous move)
        
        # Optimize H
        t0 = perf_counter()
        policy_H, x_H = self._optimize_for("H")
        t_opt_h = perf_counter() - t0
        
        # Optimize F
        t0 = perf_counter()
        policy_F, x_F = self._optimize_for("F")
        t_opt_f = perf_counter() - t0
        
        # 2. 应用决策
        # 注意：policy 包含 export_controls 和 import_tariffs
        t0 = perf_counter()
        if policy_H:
            self.sim.apply_export_control("H", policy_H.get("export", {}))
            self.sim.apply_import_tariff("H", policy_H.get("tariff", {}))
        
        if policy_F:
            self.sim.apply_export_control("F", policy_F.get("export", {}))
            self.sim.apply_import_tariff("F", policy_F.get("tariff", {}))
        t_apply = perf_counter() - t0
            
        self.history_policies["H"].append(policy_H)
        self.history_policies["F"].append(policy_F)
        self.last_x["H"] = x_H
        self.last_x["F"] = x_F
        
        # 3. 推进仿真
        t0 = perf_counter()
        self.sim.run(self.config.step_interval)
        t_sim = perf_counter() - t0
        self.round_idx += 1
        logger.info(f"--- Round {self.round_idx} End ---")

        t_total = perf_counter() - t_start
        self.timing_records.append(
            {
                "round_idx": float(self.round_idx),
                "period": float(period) if period is not None else float("nan"),
                "optimize_h_s": float(t_opt_h),
                "optimize_f_s": float(t_opt_f),
                "apply_s": float(t_apply),
                "simulate_s": float(t_sim),
                "total_s": float(t_total),
            }
        )

    def _optimize_for(self, actor: Country) -> Tuple[Dict, np.ndarray]:
        # 1. 确定目标与模式
        obj_type = self.config.objective_type_H if actor == "H" else self.config.objective_type_F
        strat_mode = self.config.strategy_mode_H if actor == "H" else self.config.strategy_mode_F
        opponent = "F" if actor == "H" else "H"
        
        # 2. 构造决策向量模版
        # 默认仅优化部分可贸易部门（演示用），可通过 config.max_sectors 调整
        x0, info = decision_vector_for_sectors(self.sim, actor=actor, max_sectors=self.config.max_sectors)
        n_x = len(x0)
        sector_idxs = info["sector_idxs"]
        K = len(sector_idxs)
        
        # 3. 确定约束 (Bounds)
        # x 结构: [m_1..m_K, tau_1..tau_K]
        # m in [0, 1], tau in [0, tau_max]
        # 对等反制逻辑：tau_self >= alpha * tau_opp_last
        
        lower_bounds = np.zeros(n_x)
        upper_bounds = np.ones(n_x) # for m part: 1.0
        
        # Expand tau bounds
        # tau 的索引为 [K, 2K)
        tau_upper = float(self.config.tau_max)
        upper_bounds[K:] = tau_upper
        
        if strat_mode == "reciprocal":
            # 获取对手上一轮的 x。如果对手上一轮没动，可能需要从 sim 状态里反推？
            # 简化：如果 last_x 存在，则用。否则认为 0。
            # 注意：last_x 是对手的 x，对应的 sector 顺序可能是一样的（如果 create_symmetric_params 且 max_sectors 一样）
            # 我们假设 顺序一致。
            opp_x = self.last_x.get(opponent)
            if opp_x is not None and len(opp_x) == n_x:
                # 对手 x 的后半部分是 taiff
                opp_tau = opp_x[K:]
                # 约束：self_tau >= alpha * opp_tau
                min_tau = self.config.reciprocal_alpha * opp_tau
                # 确保不超过 tau_upper
                min_tau = np.minimum(min_tau, tau_upper)
                
                lower_bounds[K:] = min_tau
            else:
                pass

        # 4. 定义目标函数
        def objective_wrapper(x):
            # 将 x 转为 policy dict
            p_exp, p_tar = apply_per_sector_from_x(None, x, sector_idxs=sector_idxs, tau_max=tau_upper)
            
            # SPSA 是最大化；evaluate_snapshot_objective 返回“越大越好”的得分，直接返回即可。
            score = evaluate_snapshot_objective(
                self.sim,  # 当前快照（作为 t=0 的起点）
                actor=actor,
                export_controls=p_exp,
                import_tariffs=p_tar,
                horizon=self.config.opt_horizon,
                objective_type=obj_type,
                weights=self.config.weights,
                price_scale=self.config.price_scale,
            )
            return score, {}

        def _make_start_points(x_base: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> List[np.ndarray]:
            """生成多起点初值列表（始终包含一个确定性的起点）。"""
            starts: List[np.ndarray] = []
            strategy = self.config.spsa_start_strategy
            n = max(int(self.config.spsa_multi_start), 1)

            # 统一随机数源：保证可复现（不同 actor 仍共享同一 seed，会造成相关性；需要可再分开）
            rng = np.random.default_rng(int(self.config.spsa_seed) + (0 if actor == "H" else 10_000))
            x_base = np.clip(np.asarray(x_base, dtype=float), lo, hi)

            if strategy == "neutral":
                starts.append(x_base)
                # 其余起点（若 n>1）退化为 noisy_neutral，避免全部相同
                strategy = "noisy_neutral"
                n = max(n, 2)

            if strategy == "random":
                for _ in range(n):
                    starts.append(rng.uniform(lo, hi))
            elif strategy == "noisy_neutral":
                starts.append(x_base)
                width = np.maximum(hi - lo, 1e-12)
                sigma = float(max(self.config.spsa_start_noise, 0.0))
                for _ in range(n - 1):
                    noise = rng.normal(loc=0.0, scale=sigma, size=x_base.shape) * width
                    starts.append(np.clip(x_base + noise, lo, hi))

            # 去重（防止重复起点浪费计算）
            uniq: List[np.ndarray] = []
            seen = set()
            for s in starts:
                key = tuple(np.round(s, 8).tolist())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            return uniq

        # 5. 运行 SPSA
        lb = lower_bounds
        ub = upper_bounds
        
        # 初始猜测：如果是 reciprocal，需要确保 x0 满足 lb
        x_start = np.clip(x0, lb, ub)
        start_points = _make_start_points(x_start, lb, ub)
        
        cfg = SPSAConfig(
            iterations=self.config.opt_iter,
            a0=self.config.opt_lr,        # a0：初始步长（越大越激进）
            c0=self.config.opt_perturb,   # c0：初始扰动（越大越“粗略”）
            alpha=self.config.spsa_alpha, # 步长衰减指数
            gamma=self.config.spsa_gamma, # 扰动衰减指数
            seed=self.config.spsa_seed,
        )
        
        # 5. 运行 SPSA（可选：多起点 + 多次重启）
        restarts = max(int(self.config.spsa_restarts), 1)
        best_x, best_val, best_info = None, -np.inf, {}
        for si, s0 in enumerate(start_points):
            for r in range(restarts):
                # seed 同时编码起点序号与重启序号，尽量减少相关性
                seed = int(cfg.seed) + 1_000 * si + r
                cfg_r = SPSAConfig(
                    iterations=cfg.iterations,
                    a0=cfg.a0,
                    c0=cfg.c0,
                    alpha=cfg.alpha,
                    gamma=cfg.gamma,
                    seed=seed,
                )
                x_r, val_r, info_r = spsa(
                    s0,
                    objective_wrapper,
                    lb,
                    ub,
                    cfg=cfg_r,
                )
                if val_r > best_val:
                    best_x, best_val, best_info = x_r, val_r, info_r
        
        # 6. 转回 Policy 格式
        final_exp, final_tar = apply_per_sector_from_x(None, best_x, sector_idxs=sector_idxs, tau_max=tau_upper)
        policy_res = {"export": final_exp, "tariff": final_tar}
        return policy_res, best_x
