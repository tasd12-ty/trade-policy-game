"""
博弈仿真脚本（Run Game Script）
================================
本脚本运行两国最优回应博弈仿真：
1. 初始化经济模型与仿真器
2. 预热仿真使系统稳定
3. 创建智能体并运行博弈循环
4. 输出结果统计与可视化

用法:
    python analysis/run_game.py --optimizer bo --steps 200 --interval 20 --lookahead 25

参数说明:
    --optimizer: 优化器类型 (bo/spsa/gd)
    --steps: 总仿真步数
    --interval: 决策间隔
    --lookahead: 前瞻评估步数
    --iter: 每次决策的优化迭代次数
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 确保仓库根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model.model import create_symmetric_parameters, normalize_model_params, solve_initial_equilibrium
from analysis.model.sim import TwoCountryDynamicSimulator
from analysis.game import GameSimulator, OptimalResponseAgent
# 优化器策略在 analysis/optimizers.py 中定义
from analysis.optimizers import BayesianOptimizer, SPSAOptimizer, GradientDescentOptimizer, HAS_BAYES_OPT


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行最优回应博弈仿真")
    parser.add_argument("--optimizer", type=str, default="bo", 
                        choices=["bo", "spsa", "gd"], 
                        help="优化器类型: bo(贝叶斯), spsa, gd(梯度)")
    parser.add_argument("--steps", type=int, default=200, 
                        help="总仿真步数")
    parser.add_argument("--interval", type=int, default=20, 
                        help="决策间隔（每多少步决策一次）")
    parser.add_argument("--lookahead", type=int, default=25, 
                        help="前瞻评估步数")
    parser.add_argument("--iter", type=int, default=10, 
                        help="每次决策的优化迭代次数")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # === 1. 初始化仿真环境 ===
    print(">>> 正在设置仿真环境...")
    raw_params = create_symmetric_parameters()
    eq_result = solve_initial_equilibrium(raw_params)
    
    if not eq_result["convergence_info"]["converged"]:
        print("错误: 初始均衡求解失败。")
        return

    sim = TwoCountryDynamicSimulator(
        normalize_model_params(raw_params), 
        eq_result, 
        theta_price=0.05  # 价格调整速度
    )
    
    # === 2. 预热阶段 ===
    print(">>> 预热中 (100 步)...")
    for _ in range(100):
        sim.step()
        
    # === 3. 选择优化器 ===
    optimizer_map = {
        "bo": BayesianOptimizer,
        "spsa": SPSAOptimizer,
        "gd": GradientDescentOptimizer
    }
    OptClass = optimizer_map.get(args.optimizer, BayesianOptimizer)
    if args.optimizer == "bo" and not HAS_BAYES_OPT:
        print("警告: 当前环境未安装 bayesian-optimization (bayes_opt)，已自动切换为 SPSA 优化器。")
        OptClass = SPSAOptimizer
        
    # === 4. 初始化智能体 ===
    n_sectors = sim.params.home.alpha.shape[0]
    
    # H 国智能体
    agent_h = OptimalResponseAgent(
        "H", n_sectors, 
        optimizer=OptClass(),
        lookahead_steps=args.lookahead
    )
    
    # F 国智能体（对称配置）
    agent_f = OptimalResponseAgent(
        "F", n_sectors, 
        optimizer=OptClass(),
        lookahead_steps=args.lookahead
    )
    
    # === 5. 运行博弈 ===
    print(f">>> 开始博弈 ({args.steps} 步, 间隔 {args.interval}, 优化器 {args.optimizer.upper()})...")
    
    game = GameSimulator(sim, agent_h, agent_f)
    
    t0 = time.time()
    game.run(steps=args.steps, decision_interval=args.interval)
    duration = time.time() - t0
    
    print(f">>> 博弈结束，耗时 {duration:.2f} 秒")
    
    # === 6. 分析结果 ===
    logs = pd.DataFrame(game.logs)
    
    # 分离决策日志和指标日志
    decisions = logs[logs['type'] == 'decision']
    metrics = logs[logs['type'] == 'metric']
    
    print("\n决策统计:")
    print(decisions.groupby('actor')['avg_tariff'].describe())
    
    # === 7. 可视化 ===
    output_dir = Path("analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not decisions.empty:
        plt.figure(figsize=(10, 5))
        for actor in ['H', 'F']:
            subset = decisions[decisions['actor'] == actor]
            plt.plot(subset['t'], subset['avg_tariff'], marker='o', label=f'Agent {actor}')
        plt.title(f"Average Tariff Evolution ({args.optimizer.upper()})")
        plt.xlabel("Time Step")
        plt.ylabel("Avg Tariff")
        plt.legend()
        plt.grid(True)
        
        save_path = output_dir / f"tariffs_{args.optimizer}.png"
        plt.savefig(save_path)
        print(f"图表已保存至 {save_path}")
        

if __name__ == "__main__":
    main()
