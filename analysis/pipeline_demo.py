
"""
分析流程演示 (Analysis Pipeline Demo)
-----------------------------------
本脚本演示如何利用 `analysis.model` 模块构建一个类似于 `eco_simu` 的仿真流程：
1.  设置参数并求解静态初始均衡。
2.  初始化动态仿真器。
3.  预热阶段（使系统稳定）。
4.  冲击阶段（施加关税/配额）。
5.  动态博弈阶段（模拟响应）。

更新：增加详细日志输出与图表绘制。
"""

import sys
import os
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# 确保仓库根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model.model import (
    create_symmetric_parameters,
    normalize_model_params,
    solve_initial_equilibrium,
    EPS
)
from analysis.model.sim import (
    TwoCountryDynamicSimulator,
    SimulationConfig,
    PolicyEvent
)

# 尝试导入可视化模块
try:
    from eco_simu.plotting import plot_history, plot_sector_paths, plot_diagnostics
    HAS_PLOTTING = True
except ImportError:
    print("警告: 未找到 eco_simu.plotting，将跳过绘图步骤。")
    HAS_PLOTTING = False

# 配置输出目录
OUTPUT_DIR = Path(__file__).parent / "results-1"
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_WARMUP = True  # 是否在图表中显示预热阶段（True: 显示全过程，False: 仅显示冲击后）

def run_pipeline_demo():
    print("=== 1. 设置与静态均衡 ===")
    raw_params = create_symmetric_parameters()
    
    print("正在求解初始均衡...")
    t0 = time.time()
    eq_result = solve_initial_equilibrium(raw_params)
    t_eq = time.time() - t0
    
    info = eq_result["convergence_info"]
    print(f"  收敛状态: {info['converged']} (耗时 {t_eq:.2f}s)")
    print(f"  最终残差: {info['final_residual']:.2e}")
    if not info['converged']:
        print("错误: 初始均衡未能收敛。")
        return

    print("\n=== 2. 初始化动态仿真器 ===")
    sim = TwoCountryDynamicSimulator(
        normalize_model_params(raw_params), 
        eq_result, 
        theta_price=0.05
    )
    print(f"  仿真计算设备: {sim.home_state.device}")

    # --- 数据记录器 ---
    logs = []

    def log_step(period, phase, action=""):
        h = sim.home_state
        f = sim.foreign_state
        # 提取关键指标
        rec = {
            "Period": period,
            "Phase": phase,
            "Event": action,
            "H_Income": h.income.item(),
            "F_Income": f.income.item(),
            "H_Price_Mean": h.price.mean().item(),
            "F_Price_Mean": f.price.mean().item(),
            "Trade_Balance_H": (h.export_actual * h.price).sum().item() - \
                               (h.imp_price * (h.X_imp.sum(dim=0) + h.C_imp)).sum().item()
        }
        logs.append(rec)

    # 3. 预热
    print(f"\n=== 3. 预热阶段 (1000 步, 显示预热: {SHOW_WARMUP}) ===")
    for t in range(1000):
        sim.step()
        log_step(t - 1000, "Warmup")
        if t % 100 == 0:
            print(f"  [预热] 步 {t}: H均价={logs[-1]['H_Price_Mean']:.4f}")
            
    # 根据配置决定是否清除历史
    if not SHOW_WARMUP:
        # 清除预热历史，仅保留当前状态作为新起点
        warmup_final_H = sim.home_state.detach()
        warmup_final_F = sim.foreign_state.detach()
        sim.history = {"H": [warmup_final_H], "F": [warmup_final_F]}
        print("  [预热完成] 历史记录已清除 (SHOW_WARMUP=False)，准备开始主仿真。")
    else:
        print("  [预热完成] 保留历史记录 (SHOW_WARMUP=True)，准备开始主仿真。")

    # 4. 冲击
    print("\n=== 4. 政策冲击 ===")
    shock_note = "H对F部门2/3加征20%关税"
    print(f"  动作: {shock_note}")
    
    # 记录决策
    log_step(0, "Shock", shock_note)
    
    shock_sectors = {2: 0.2, 3: 0.2} 
    # sim.apply_import_tariff("H", shock_sectors, note=shock_note)
    
    # 5. 动态仿真
    print("\n=== 5. 动态仿真 (1000 步) ===")
    for t in range(1, 1001):
        sim.step()
        log_step(t, "Dynamics")
        
        # 每 5 步详细打印
        if t % 100 == 0:
            last = logs[-1]
            print(f"  [动态] 步 {t:02d}: H收入={last['H_Income']:.4f}, "
                  f"F收入={last['F_Income']:.4f}, "
                  f"H贸易余额={last['Trade_Balance_H']:.4f}")

    # --- 6. 结果汇总与可视化 ---
    print("\n=== 6. 结果输出 ===")
    
    # 保存日志 CSV
    df_log = pd.DataFrame(logs)
    csv_path = OUTPUT_DIR / "demo_logs.csv"
    df_log.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"  详细日志已保存: {csv_path}")
    
    # 打印最后几行日志
    print("\n  [日志摘要 - 最后 5 步]")
    print(df_log.tail(5).to_string(index=False))

    if HAS_PLOTTING:
        print("\n  正在生成图表...")
        try:
            from eco_simu.plotting import plot_history_with_events
            
            # 构建用于绘图的事件列表
            plot_events = []
            # 判断是否有实际冲击
            has_real_shock = any(v > 0 for v in shock_sectors.values())
            
            if has_real_shock:
                sector_str = ",".join(f"{k}:{int(v*100)}%" for k, v in shock_sectors.items())
                # 如果保留预热历史，事件时间需要加上预热步数
                t0 = 1000 if SHOW_WARMUP else 0
                
                plot_events.append({
                    "t": t0,
                    "actor": "H",
                    "action": f"Import Tariff [{sector_str}]"
                })
                plot_events.append({
                    "t": t0,
                    "actor": "F",
                    "action": "No Action"
                })
            else:
                t0 = 1000 if SHOW_WARMUP else 0
                plot_events.append({
                    "t": t0,
                    "actor": "H",
                    "action": "No Action"
                })
                plot_events.append({
                    "t": t0,
                    "actor": "F",
                    "action": "No Action"
                })
            
            # 带事件标注的历史图
            plot_history_with_events(
                sim, 
                policy_events=plot_events,
                save_path=str(OUTPUT_DIR / "history_with_events.png"), 
                show=False,
                warmup=0,  # 不裁剪，从冲击开始展示
                title=f"Two-Country Simulation - Policy Events (Warmup={'Shown' if SHOW_WARMUP else 'Hidden'})"
            )
            
            # 诊断面板
            plot_diagnostics(
                sim, 
                save_path=str(OUTPUT_DIR / "diagnostics.png"), 
                show=False, 
                warmup=0
            )
            
            # 部门路径 (所有部门，绝对价格)
            plot_sector_paths(
                sim, "H", "price", sectors=None,  # None = 所有部门
                save_path=str(OUTPUT_DIR / "sector_price_H.png"),
                show=False, warmup=0, relative=False  # relative=False 显示绝对价格
            )
            plot_sector_paths(
                sim, "F", "price", sectors=None,
                save_path=str(OUTPUT_DIR / "sector_price_F.png"),
                show=False, warmup=0, relative=False
            )
             
            print(f"  图表已保存至: {OUTPUT_DIR}")
        except Exception as e:
            import traceback
            print(f"  绘图失败: {e}")
            traceback.print_exc()
    else:
        print("  跳过绘图 (缺少 plotting 模块)")

    print("\n演示结束。")

if __name__ == "__main__":
    run_pipeline_demo()
