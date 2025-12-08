from __future__ import annotations

import os
import numpy as np

from eco_simu import SimulationConfig, ConflictBlock, PolicyEvent, simulate
from eco_simu import create_symmetric_parameters


def main():
    # 配置：总期数1000，第500期开始提高两个部门关税20%
    # 假设：对 Home 国的部门2与3（0基索引）加征20%；如需调整可修改下方映射。
    # 互征关税（不同部门）：
    # Home 对部门 2、3 加征 20%，Foreign 对部门 0、4 加征 20%
    conflict = ConflictBlock(
        export_controls={
            "H": {1: 1}  # 0.8 表示对部门1出口限额为80%
        },
        import_tariffs={
            "H": {2: 0, 3: 0},
            "F": {0: 0, 4: 0},
        },
    )

    # 事件时间线
    events = [
        # PolicyEvent(kind="import_tariff", actor="H", sectors={2: 0.2}, start_period=220),
        # PolicyEvent(kind="import_tariff", actor="H", sectors={2: 0.4}, start_period=250),
        # PolicyEvent(kind="export_quota", actor="H", sectors={2: 0.8}, start_period=300),
        # PolicyEvent(kind="import_tariff", actor="H", sectors={2: 0.1}, start_period=500),
        # # 示例：Home 对部门2征收 15% 出口税，从 t=350 到 t=480（事件翻译为 Foreign 的进口关税）
        # PolicyEvent(kind="export_tariff", actor="H", sectors={2: 0.15}, start_period=350, end_period=480),
    ]
    config = SimulationConfig(
        total_periods=1000,
        conflict_start=200,
        theta_price=0.05,
        solver_max_iter=400,
        solver_tol=1e-8,
        events=events ,
        conflict=conflict,
    )

    params = create_symmetric_parameters()
    sim = simulate(config, params_raw=params)

    # 汇总并保存结果
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

    # 生成图表
    try:
        # 仅调用 eco_simu 内的绘图方法；移除对已迁移模块的多余依赖
        sim.plot_history(save_path=os.path.join(outdir, 'dynamic_history.png'), show=False)
        sim.plot_sector_paths('H', 'output', save_path=os.path.join(outdir, 'sector_output_H.png'), show=False, relative=True)
        sim.plot_sector_paths('F', 'output', save_path=os.path.join(outdir, 'sector_output_F.png'), show=False, relative=True)
        sim.plot_sector_paths('H', 'price', save_path=os.path.join(outdir, 'sector_price_H.png'), show=False, relative=True)
        sim.plot_sector_paths('F', 'price', save_path=os.path.join(outdir, 'sector_price_F.png'), show=False, relative=True)
        sim.plot_diagnostics(save_path=os.path.join(outdir, 'model_diagnostics.png'), show=False)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("Simulation done: 1000 periods, conflict at t=500 (H:2&3 +20%, F:0&4 +20%)")
    print(f"Home final income growth: {summary['H']['income_growth'][-1]:.2f}%")
    print(f"Foreign final income growth: {summary['F']['income_growth'][-1]:.2f}%")
    print("Outputs saved under results/: history_*.csv and PNG plots")


if __name__ == "__main__":
    main()
