"""
优化器基准测试脚本（Compare Optimizers Script）
================================================
本脚本对比不同优化策略在单次最优回应问题上的性能：
- 贝叶斯优化 (BO)
- 同时扰动随机逼近 (SPSA)
- 数值梯度上升 (GD)

输出:
- 各优化器的最优值与耗时
- 收敛曲线对比图

用法:
    python analysis/compare_optimizers.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import copy

# 确保仓库根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model.model import create_symmetric_parameters, normalize_model_params, solve_initial_equilibrium
from analysis.model.sim import TwoCountryDynamicSimulator
from analysis.game import OptimalResponseAgent
# 优化器策略在 analysis/optimizers.py 中定义
from analysis.optimizers import BayesianOptimizer, SPSAOptimizer, GradientDescentOptimizer, HAS_BAYES_OPT


def run_benchmark():
    """运行优化器基准测试。"""
    print(">>> 正在设置基准测试环境...")
    
    # === 1. 初始化仿真环境 ===
    raw_params = create_symmetric_parameters()
    eq_result = solve_initial_equilibrium(raw_params)
    sim = TwoCountryDynamicSimulator(
        normalize_model_params(raw_params), 
        eq_result
    )
    
    # 轻度预热
    for _ in range(10): 
        sim.step()
        
    # === 2. 定义测试问题 ===
    n_sectors = sim.params.home.alpha.shape[0]
    print(f">>> 问题规模: N={n_sectors} 部门")
    
    # 初始化待测试的优化器（BO 为可选依赖）
    optimizers = {
        "SPSA": SPSAOptimizer(),
        "GD": GradientDescentOptimizer(),
    }
    if HAS_BAYES_OPT:
        optimizers["BO"] = BayesianOptimizer()
    else:
        print("警告: 当前环境未安装 bayesian-optimization (bayes_opt)，已跳过 BO 基准测试。")
    
    # 创建基准智能体实例（用于参考）
    agent_base = OptimalResponseAgent("H", n_sectors, lookahead_steps=10)
    
    # 定义目标函数工厂（简化版：仅最大化收入增长）
    def make_objective(sim_snapshot, lookahead: int = 10):
        """
        创建目标函数。
        
        参数:
            sim_snapshot: 仿真器快照
            lookahead: 前瞻步数
        
        返回:
            objective: 目标函数，输入参数向量，返回收入增长
        """
        def objective(params):
            # 解码参数: [关税_0, ..., 关税_n, 乘子_0, ..., 乘子_n]
            n = sim.params.home.alpha.shape[0]
            
            tariffs = params[:n]
            multipliers = params[n:]
            
            # 构建行动
            action = {
                "import_tariff": {i: t for i, t in enumerate(tariffs) if t > 1e-4},
                "export_quota": {i: m for i, m in enumerate(multipliers) if abs(m - 1.0) > 1e-4}
            }
            
            # 克隆仿真器
            sim_run = copy.deepcopy(sim_snapshot)
            sim_run.apply_action("H", action)
            
            # 运行前瞻仿真，计算收入增长
            init_income = sim_run.home_state.income.item()
            for _ in range(lookahead):
                sim_run.step()
                
            final_income = sim_run.home_state.income.item()
            return final_income - init_income  # 收入增量
            
        return objective

    # 创建目标函数实例
    obj_func = make_objective(sim, lookahead=5)
    
    # 构建参数边界
    # N 个关税 ∈ [0, 0.5], N 个乘子 ∈ [0.5, 1.0]
    n = sim.params.home.alpha.shape[0]
    bounds = []
    for _ in range(n): 
        bounds.append((0.0, 0.5))
    for _ in range(n): 
        bounds.append((0.5, 1.0))
    bounds_arr = np.array(bounds)
    
    # 初始猜测：中性政策
    init_guess = np.array([0.0]*n + [1.0]*n)
    
    # === 3. 运行基准测试 ===
    results = {}
    
    print(">>> 运行基准测试...")
    for name, opt in optimizers.items():
        print(f"  正在测试 {name}...")
        t0 = time.time()
        best_p, best_val, hist = opt.optimize(
            obj_func, 
            bounds_arr, 
            initial_guess=init_guess, 
            n_iter=20  # 固定迭代预算
        )
        duration = time.time() - t0
        
        results[name] = {
            "best_val": best_val,
            "time": duration,
            "history": hist
        }
        print(f"    {name}: 最优值={best_val:.4f}, 耗时={duration:.2f}s")
        
    # === 4. 绘制对比图 ===
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['history'], label=f"{name} (Final: {res['best_val']:.4f})")
        
    plt.title("Optimizer Convergence Comparison (Objective: Income Growth)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    out_path = Path("analysis/results/optimizer_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\n>>> 对比图已保存至 {out_path}")


if __name__ == "__main__":
    run_benchmark()
