"""对比博弈场景运行脚本

运行多种博弈场景并对比结果：
1. Baseline - 无额外政策的基线
2. Standard - 双方都追求自身福利最大化
3. Relative - 双方都追求相对优势（零和博弈）
4. Reciprocal - 双方采用对等反制策略

生成可视化对比图：
- 收入增长曲线
- 贸易差额变化
- 关税政策轨迹
- 价格指数变化
"""

import sys
from pathlib import Path
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import logging

# 兼容直接执行：将仓库根目录加入 sys.path，确保可导入 analysis 命名空间
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from analysis.optimization.interaction import InteractiveGame, GameConfig
from analysis.model import (bootstrap_simulator, create_symmetric_parameters)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 实验控制参数（固定热启动期数 + 可选隐藏热启动）
# -------------------------
# 说明：
# - 为了让曲线更“干净”，默认将热启动阶段从可视化中移除（SHOW_WARMUP=False）。
# - 热启动用于让模型从静态均衡过渡到其动态稳态附近，减少前几期的非政策波动。
WARMUP_PERIODS = 200
SHOW_WARMUP = False


def _reset_history(sim):
    """将历史重置为“当前状态作为 t=0”，用于隐藏热启动阶段。"""
    h0 = sim.home_state.detach()
    f0 = sim.foreign_state.detach()
    sim.history = {"H": [h0], "F": [f0]}
    # 事件日志也从新起点开始，避免图表标注混入热启动阶段
    sim.policy_events = []


def run_scenario(name, config, steps=20, shock=None, *, warmup_periods: int = 0, show_warmup: bool = True):
    logger.info(f"Starting Scenario: {name}")
    # 初始化新的仿真器（避免不同场景之间状态互相污染）
    params = create_symmetric_parameters()
    sim = bootstrap_simulator(params)
    if warmup_periods > 0:
        sim.run(int(warmup_periods))  # 热启动（warmup）

    # 可选：施加外生冲击，让“回应策略”更有意义（否则对称环境下可能都趋向于不动）
    if shock:
        country = shock.get("country", "H")
        tariff = shock.get("tariff", None)
        export_m = shock.get("export", None)
        note = shock.get("note", "Exogenous Shock")
        if tariff:
            sim.apply_import_tariff(country, tariff, note=note)
        if export_m:
            sim.apply_export_control(country, export_m, note=note)

    # 如果不显示热启动，则把“热启动 + 外生冲击”后的状态作为新的 t=0
    if not show_warmup:
        _reset_history(sim)
    
    game = InteractiveGame(config, initial_sim=sim)
    
    # 运行博弈
    for i in range(steps):
        game.step()
        
    return game.sim.summarize_history(), game.history_policies, game.timing_records

def main():
    steps = 100  # 演示用：跑 5 轮决策
    interval = 10

    # 外生冲击：H 先对可贸易部门加征关税，让三类策略有可比较的“响应对象”
    shock = {
        "country": "H",
        "tariff": {2: 0.2, 3: 0.2},
        "export": None,
        "note": "Initial shock: H tariff +20% on sectors 2&3",
    }
    
    scenarios = {
        "Baseline": None,  # 单独处理
        "Standard": GameConfig(
            objective_type_H="standard", objective_type_F="standard",
            strategy_mode_H="independent", strategy_mode_F="independent",
            step_interval=interval, opt_iter=120, opt_horizon=15,
            opt_lr=0.18, opt_perturb=0.04,
            # 降低价格项放大倍数，避免“过度维稳 -> 都不动”的无聊策略
            price_scale=1000.0,
            # 增加可优化部门数量，让策略空间更丰富（同时会更慢）
            max_sectors=2,
            spsa_multi_start=8,                 # 起点个数（例如 4）
            spsa_start_strategy="random",       # 每个起点在边界内均匀采样
            spsa_restarts=1,                    # 每个起点跑 1 次（可再加大）
            spsa_seed=42,
        ),
        # "Relative": GameConfig(
        #     objective_type_H="relative", objective_type_F="relative",
        #     strategy_mode_H="independent", strategy_mode_F="independent",
        #     step_interval=interval, opt_iter=50, opt_horizon=10,
        #     opt_lr=0.05, opt_perturb=0.1,
        #     price_scale=500.0,
        #     max_sectors=4,
        # ),
        # "Reciprocal": GameConfig(
        #     objective_type_H="standard", objective_type_F="standard",
        #     strategy_mode_H="reciprocal", strategy_mode_F="reciprocal",
        #     reciprocal_alpha=1.0,
        #     step_interval=interval, opt_iter=50, opt_horizon=10,
        #     opt_lr=0.05, opt_perturb=0.1,
        #     price_scale=500.0,
        #     max_sectors=4,
        # ),
    }
    
    results = {}
    policy_records = {}  # 存放策略历史（用于标注/对比）
    
    # Run Baseline
    logger.info("Running Baseline...")
    params = create_symmetric_parameters()
    sim_base = bootstrap_simulator(params)
    if WARMUP_PERIODS > 0:
        sim_base.run(int(WARMUP_PERIODS))
    # 基线也施加相同外生冲击，但不进行策略优化（保持“无回应”）
    if shock.get("tariff"):
        sim_base.apply_import_tariff(shock.get("country", "H"), shock["tariff"], note=shock.get("note"))
    if shock.get("export"):
        sim_base.apply_export_control(shock.get("country", "H"), shock["export"], note=shock.get("note"))
    if not SHOW_WARMUP:
        _reset_history(sim_base)
    # 推进与其它场景等长的时间（steps * interval）
    sim_base.run(steps * interval)
    results["Baseline"] = sim_base.summarize_history()
    policy_records["Baseline"] = {"H": [], "F": []}  # 基线无策略

    # Run others
    for name, conf in scenarios.items():
        if name == "Baseline": continue
        try:
            res_hist, res_pol, res_timing = run_scenario(
                name,
                conf,
                steps=steps,
                shock=shock,
                warmup_periods=WARMUP_PERIODS,
                show_warmup=SHOW_WARMUP,
            )
            results[name] = res_hist
            policy_records[name] = res_pol
            policy_records[name]["_timing"] = res_timing
        except Exception as e:
            logger.error(f"Scenario {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # 时间统计输出与保存
    _export_timing_csv(policy_records, steps=steps, interval=interval)

    # Plotting
    plot_comparisons(results, policy_records, steps, interval, warmup_periods=WARMUP_PERIODS, show_warmup=SHOW_WARMUP, shock=shock)

def _export_timing_csv(policy_records: dict, *, steps: int, interval: int) -> None:
    """汇总并导出交互博弈的时间统计。

    - 控制台打印每个场景的平均耗时；
    - 保存逐轮明细到 analysis/results/ 下的 CSV。
    """
    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"timing_run_comparative_game_{ts}.csv"

    rows = []
    for scen, pol in policy_records.items():
        timing = pol.get("_timing", [])
        if not timing:
            continue
        # 简单汇总：平均每轮耗时、优化耗时占比等
        totals = [r.get("total_s", 0.0) for r in timing]
        opt = [r.get("optimize_h_s", 0.0) + r.get("optimize_f_s", 0.0) for r in timing]
        sim = [r.get("simulate_s", 0.0) for r in timing]
        avg_total = float(np.mean(totals)) if totals else 0.0
        avg_opt = float(np.mean(opt)) if opt else 0.0
        avg_sim = float(np.mean(sim)) if sim else 0.0
        logger.info(
            f"[Timing] {scen}: rounds={len(timing)}, interval={interval}, "
            f"avg_total={avg_total:.3f}s, avg_opt={avg_opt:.3f}s, avg_sim={avg_sim:.3f}s"
        )
        for r in timing:
            rows.append({"scenario": scen, **r})

    if not rows:
        logger.info("[Timing] No timing records to export.")
        return

    # 写 CSV（避免引入额外依赖）
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"[Timing] Saved timing CSV to {out_csv}")

def _format_mapping(mapping: dict, label: str) -> str:
    """将部门策略字典格式化为英文短串，例如 tariff{2:0.10,3:0.05}。"""
    if not mapping:
        return f"{label}{{}}"
    parts = [f"{int(k)}:{float(v):.2f}" for k, v in mapping.items()]
    return f"{label}{{" + ",".join(parts) + "}}"

def _policy_to_text(policy: dict) -> str:
    """将单方策略 dict 转为英文文本。"""
    if not policy:
        return "tariff{} export{}"
    tar = policy.get("tariff", {}) or {}
    exp = policy.get("export", {}) or {}
    return f"{_format_mapping(tar, 'tariff')} {_format_mapping(exp, 'export_m')}"

def plot_comparisons(results, policies, steps, interval, *, warmup_periods: int, show_warmup: bool, shock: dict):
    # 增加一个“决策时间线”面板，用于展示每轮 H/F 的决策文本，避免在数据曲线上过度拥挤
    fig, axes = plt.subplots(
        5, 1,
        figsize=(12, 23),
        gridspec_kw={"height_ratios": [4, 4, 4, 4, 2]},
        sharex=False
    )
    
    colors = {"Baseline": "gray", "Standard": "blue", "Relative": "red", "Reciprocal": "green"}
    
    # 1. Income Comparison (H)
    ax = axes[0]
    for name, res in results.items():
        inc = res["H"]["income"]
        # 如果不显示热启动，history 已经被重置，t=0 就是热启动后的起点
        x = np.arange(len(inc), dtype=int)
        inc_norm = inc / inc[0]
        ax.plot(x, inc_norm, label=name, color=colors.get(name, "black"))
    ax.set_title("Income Growth (H)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Trade Balance (H)
    ax = axes[1]
    for name, res in results.items():
        tb = res["H"]["trade_balance_val"]
        x = np.arange(len(tb), dtype=int)
        ax.plot(x, tb, label=name, color=colors.get(name, "black"))
    ax.set_title("Trade Balance (H)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Import Tariff Policy (avg)
    # 从策略记录中提取平均关税
    ax = axes[2]
    # 如果 SHOW_WARMUP=False，我们已将热启动阶段剥离，因此 burn_in=0
    burn_in = int(warmup_periods) if show_warmup else 0
    
    # 决策发生的时间点（与 InteractiveGame.step() 的应用时刻一致）
    decision_times = [burn_in + k * interval for k in range(steps)]
    
    for name, pol_dict in policies.items():
        if name == "Baseline":
            continue
        
        # pol_dict["H"] 是每轮策略字典列表：{'export': {sector: m}, 'tariff': {sector: tau}}
        # 为绘图构造阶梯函数时间序列
        h_pols = pol_dict["H"]
        
        # 收集每轮的平均关税（对该轮涉及的部门求均值）
        tariff_trajectory = []
        time_points = []
        
        current_tau = 0.0
        # t=0 到 burn_in：视为 0
        tariff_trajectory.append(0.0)
        time_points.append(0)
        tariff_trajectory.append(0.0)
        time_points.append(burn_in)
        
        for k, p in enumerate(h_pols):
            # 更新当前轮的平均关税
            if p and "tariff" in p and p["tariff"]:
                taus = list(p["tariff"].values())
                current_tau = sum(taus) / len(taus)
            
            # 画阶梯：该轮值保持到下一轮
            start_t = burn_in + k * interval
            end_t = burn_in + (k+1) * interval
            
            # 添加阶梯点
            tariff_trajectory.append(current_tau)
            time_points.append(start_t)
            tariff_trajectory.append(current_tau)
            time_points.append(end_t)

        ax.plot(time_points, tariff_trajectory, label=name, color=colors[name], linewidth=2)
        
    ax.set_title("Avg Import Tariff Level (H Policy)")
    ax.set_ylabel("Tariff Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 在四个数据面板上标注决策时间（垂直虚线，英文图上不使用中文）
    for t in decision_times:
        for a in axes[:4]:
            a.axvline(t, color="gray", linestyle=":", alpha=0.35)
    # 外生冲击标注（t=burn_in，若隐藏热启动则 t=0）
    shock_t = burn_in
    for a in axes[:4]:
        a.axvline(shock_t, color="black", linestyle="--", alpha=0.5)

    # 4. Price Index (H)
    ax = axes[3]
    for name, res in results.items():
        p = res["H"]["price_mean"]
        x = np.arange(len(p), dtype=int)
        p_rel = p / p[0]
        ax.plot(x, p_rel, label=name, color=colors.get(name, "black"))
    ax.set_title("Price Index (H) - Inflation/Deflation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 决策信息面板（英文）
    ax_text = axes[4]
    ax_text.axis("off")
    shock_desc = shock.get("note", "Shock")
    text_lines = [
        "Policy Decisions (per round):",
        f"Shock @t={shock_t}: {shock_desc}",
    ]
    # 添加每个场景的耗时概览（英文）
    for scen_name, pol_dict in policies.items():
        timing = pol_dict.get("_timing", [])
        if not timing:
            continue
        totals = [r.get("total_s", 0.0) for r in timing]
        opt = [r.get("optimize_h_s", 0.0) + r.get("optimize_f_s", 0.0) for r in timing]
        avg_total = float(np.mean(totals)) if totals else 0.0
        avg_opt = float(np.mean(opt)) if opt else 0.0
        text_lines.append(f"Timing {scen_name}: avg_step={avg_total:.3f}s, avg_opt={avg_opt:.3f}s")
    for scen_name, pol_dict in policies.items():
        if scen_name == "Baseline":
            continue
        h_pols = pol_dict.get("H", [])
        f_pols = pol_dict.get("F", [])
        for k in range(steps):
            t = burn_in + k * interval
            pH = h_pols[k] if k < len(h_pols) else {}
            pF = f_pols[k] if k < len(f_pols) else {}
            line = (
                f"{scen_name} R{k+1} @t={t}: "
                f"H {_policy_to_text(pH)}; "
                f"F {_policy_to_text(pF)}"
            )
            text_lines.append(line)
    ax_text.text(
        0.01, 0.98,
        "\n".join(text_lines),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        color="black",
    )
    
    plt.tight_layout()
    plt.savefig("scenario_comparison.png")
    logger.info("Saved plot to scenario_comparison.png")

if __name__ == "__main__":
    main()
