from __future__ import annotations

import os
import numpy as np

from eco_simu import SimulationConfig, ConflictBlock, PolicyEvent, simulate, bootstrap_simulator
from eco_simu import create_symmetric_parameters

# 预热配置
WARMUP_PERIODS = 1000  # 预热步数
SIMULATION_PERIODS = 1000  # 主仿真步数
SHOW_WARMUP = False  # 是否在图表中显示预热阶段（True: 显示全过程，False: 仅显示冲击后）


def main():
    # ===== 1. 预热阶段 =====
    print(f"=== 预热阶段 (显示预热: {SHOW_WARMUP}) ===")
    params = create_symmetric_parameters()
    sim = bootstrap_simulator(params_raw=params, theta_price=0.05)
    
    # 运行预热
    for t in range(WARMUP_PERIODS):
        sim.step()
        if t % 100 == 0:
            print(f"  预热步 {t}: H均价={sim.home_state.price.mean().item():.4f}")
    
    # 根据配置决定是否清除历史
    if not SHOW_WARMUP:
        # 清除预热历史，仅保留当前状态作为新起点
        warmup_final_H = sim.home_state.detach()
        warmup_final_F = sim.foreign_state.detach()
        sim.history = {"H": [warmup_final_H], "F": [warmup_final_F]}
        print(f"  预热完成，历史记录已清除 (SHOW_WARMUP=False)\n")
    else:
        print(f"  预热完成，保留历史记录 (SHOW_WARMUP=True)\n")

    # ===== 2. 冲击配置 =====
    # 定义政策事件时间线（相对于主仿真开始的时刻）
    # 格式: (触发时刻, 动作类型, 国家, 部门映射, 备注)
    policy_timeline = [
        # 示例1: t=0 时立即施加冲击（冲突开始）
        (0, "import_tariff", "H", {2: 0.2, 3: 0.2}, "H对部门2/3加征20%关税"),
    ]
    
    # ===== 3. 主仿真阶段（带政策事件） =====
    print(f"=== 主仿真阶段 ({SIMULATION_PERIODS} 步) ===")
    
    # 将事件时间线转为字典便于查找
    event_schedule = {}
    for (trigger_t, action, actor, sectors, note) in policy_timeline:
        event_schedule.setdefault(trigger_t, []).append((action, actor, sectors, note))
    
    for t in range(SIMULATION_PERIODS):
        # 检查是否有事件需要在此时刻触发
        if t in event_schedule:
            for (action, actor, sectors, note) in event_schedule[t]:
                if action == "import_tariff":
                    sim.apply_import_tariff(actor, sectors, note=note)
                    print(f"  [事件 t={t}] {note}")
                elif action == "export_quota":
                    sim.apply_export_control(actor, sectors, note=note)
                    print(f"  [事件 t={t}] {note}")
                elif action == "import_multiplier":
                    sim.set_import_multiplier(actor, sectors, relative_to_baseline=True, note=note)
                    print(f"  [事件 t={t}] {note}")
        
        # 执行仿真步
        sim.step()
        
        if t % 100 == 0:
            print(f"  仿真步 {t}: H收入={sim.home_state.income.item():.4f}")

    # ===== 4. 汇总并保存结果 =====
    summary = sim.summarize_history()
    periods = np.arange(len(sim.history["H"]))

    outdir = os.path.join("results")
    os.makedirs(outdir, exist_ok=True)

    def save_country(tag: str):
        data = summary[tag]
        arr = np.column_stack([
            periods,
            data["income"],
            data["output_sum"],
            data["export_value"],
            data["import_value"],
            data["trade_balance"],
            data["import_value_val"],
            data["trade_balance_val"],
            data["income_growth"],
            data["output_growth"],
        ])
        header = (
            "period,income,output_sum,export_value,import_value,trade_balance,"
            "import_value_val,trade_balance_val,income_growth,output_growth"
        )
        np.savetxt(os.path.join(outdir, f"history_{tag}.csv"), arr, delimiter=",", header=header, comments="")

    save_country("H")
    save_country("F")

    # 生成图表（带政策事件标注）
    try:
        from eco_simu.plotting import plot_history_with_events
        
        # 构建用于绘图的事件列表
        plot_events = []
        if policy_timeline:
            for (trigger_t, action, actor, sectors, note) in policy_timeline:
                # 如果保留预热历史，事件时间需要加上预热步数
                display_t = trigger_t + WARMUP_PERIODS if SHOW_WARMUP else trigger_t
                
                sector_str = ",".join(f"{k}:{int(v*100)}%" for k, v in sectors.items())
                plot_events.append({
                    "t": display_t,
                    "actor": actor,
                    "action": f"{action.replace('_', ' ')} [{sector_str}]"
                })
        else:
            # 无事件时也标注
            t0 = WARMUP_PERIODS if SHOW_WARMUP else 0
            plot_events.append({"t": t0, "actor": "H/F", "action": "No Policy Change"})
        
        plot_history_with_events(
            sim, 
            policy_events=plot_events,
            save_path=os.path.join(outdir, 'history_with_events.png'), 
            show=False,
            title=f"Two-Country Simulation - Policy Events (Warmup={'Shown' if SHOW_WARMUP else 'Hidden'})"
        )
        
        # 其他图表
        sim.plot_sector_paths('H', 'output', save_path=os.path.join(outdir, 'sector_output_H.png'), show=False, relative=True)
        sim.plot_sector_paths('F', 'output', save_path=os.path.join(outdir, 'sector_output_F.png'), show=False, relative=True)
        sim.plot_diagnostics(save_path=os.path.join(outdir, 'model_diagnostics.png'), show=False)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"\n=== 仿真完成 ===")
    print(f"  预热: {WARMUP_PERIODS} 步, 主仿真: {SIMULATION_PERIODS} 步")
    print(f"  政策事件: {len(policy_timeline)} 个")
    print(f"  Home 最终收入增长率: {summary['H']['income_growth'][-1]:.2f}%")
    print(f"  Foreign 最终收入增长率: {summary['F']['income_growth'][-1]:.2f}%")
    print(f"  结果已保存至: results/")


if __name__ == "__main__":
    main()
