"""可选绘图模块（迁移至 EcoModel 包）。"""
# Moved from EcoModel to eco_simu package.

from __future__ import annotations

from typing import List, Optional

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def _configure_chinese_fonts() -> None:
    candidates = [
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'Source Han Sans SC',
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Sarasa UI SC', 'Arial Unicode MS',
    ]
    for font in candidates:
        try:
            fm.findfont(font, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams['font.family'] = font
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        return
    plt.rcParams['axes.unicode_minus'] = False


_configure_chinese_fonts()


def _safe_rebase_growth(series: np.ndarray) -> np.ndarray:
    if series.size == 0:
        return series
    base = float(series[0])
    if abs(base) < 1e-12:
        return np.zeros_like(series)
    return ((series / base) - 1.0) * 100.0


def _trimmed_summary(summary: dict, trim: int) -> dict:
    trim_val = max(int(trim or 0), 0)
    result: dict = {}
    for country in ("H", "F"):
        src = summary.get(country, {}) or {}
        data_c: dict = {}
        for key, val in src.items():
            arr = np.array(val, float)
            data_c[key] = arr[trim_val:] if trim_val else arr
        # 重新以裁剪后首期为基准计算增长率
        data_c["income_growth"] = _safe_rebase_growth(np.array(data_c.get("income", []), float))
        data_c["output_growth"] = _safe_rebase_growth(np.array(data_c.get("output_sum", []), float))
        result[country] = data_c
    return result


def plot_history(
    sim,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    warmup: int = 0,
) -> None:
    """绘制两国宏观历史；可选 warmup>0 时裁掉前若干期并重新以 0 起算。

    若需要以“智能体开始博弈”为起点并在每回合标注决策，请使用
    `plot_history_agent_view(...)`。
    """
    metrics = metrics or ["income", "output_sum", "trade_balance", "income_growth"]
    summary = sim.summarize_history()
    total_len = len(sim.history["H"])
    trim = max(int(warmup or 0), 0)
    if total_len > 0 and trim >= total_len:
        trim = max(total_len - 1, 0)
    periods = np.arange(total_len - trim)
    data = _trimmed_summary(summary, trim)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = {'H': '#1f77b4', 'F': '#ff7f0e'}
    linestyles = {'H': '-', 'F': '--'}

    for ax, metric in zip(axes, metrics):
        if metric not in data["H"]:
            continue
        ax.plot(periods, data["H"][metric], color=colors['H'], linestyle=linestyles['H'],
                linewidth=2, marker='o', markersize=2, label="Home")
        ax.plot(periods, data["F"][metric], color=colors['F'], linestyle=linestyles['F'],
                linewidth=2, marker='s', markersize=2, label="Foreign")
        ylabel = metric.replace("_", " ").title()
        if "growth" in metric:
            ylabel += " (%)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        if "growth" in metric:
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel("Periods", fontsize=10)
    fig.suptitle("Two-Country Economic Simulation", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.subplots_adjust(top=0.93, bottom=0.08)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sector_paths(sim, country: str, metric: str, sectors: Optional[List[int]] = None,
                      save_path: Optional[str] = None, show: bool = False, relative: bool = True,
                      warmup: int = 0) -> None:
    country = country.upper()
    states_all = sim.history[country]
    trim = max(int(warmup or 0), 0)
    if states_all:
        if trim >= len(states_all):
            trim = max(len(states_all) - 1, 0)
    states = states_all[trim:]
    if not states:
        return
    series = None
    if metric == "price":
        series = np.vstack([s.price.detach().cpu().numpy() for s in states])
    elif metric == "output":
        series = np.vstack([s.output.detach().cpu().numpy() for s in states])
    elif metric == "export":
        series = np.vstack([s.export_actual.detach().cpu().numpy() for s in states])
    elif metric == "consumption_domestic":
        series = np.vstack([s.C_dom.detach().cpu().numpy() for s in states])
    elif metric == "consumption_import":
        series = np.vstack([s.C_imp.detach().cpu().numpy() for s in states])
    elif metric == "intermediate_domestic":
        series = np.vstack([s.X_dom.sum(dim=0).detach().cpu().numpy() for s in states])
    elif metric == "intermediate_import":
        series = np.vstack([s.X_imp.sum(dim=0).detach().cpu().numpy() for s in states])
    else:
        raise ValueError(f"不支持的指标: {metric}")

    periods = np.arange(series.shape[0])
    sectors = sectors or list(range(series.shape[1]))
    if relative and series.shape[0] > 1:
        display_series = np.zeros_like(series)
        for i in range(series.shape[1]):
            display_series[:, i] = ((series[:, i] / max(series[0, i], 1e-9)) - 1) * 100 if series[0, i] > 1e-9 else 0
        ylabel_suffix = " - Relative Change (%)"
    else:
        display_series = series
        ylabel_suffix = ""

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    for i, sec in enumerate(sectors):
        total_change = abs(display_series[-1, sec] - display_series[0, sec])
        alpha = 1.0 if total_change > np.std(display_series[:, sectors]) else 0.6
        ax.plot(periods, display_series[:, sec], color=colors[i], linewidth=2, alpha=alpha,
                marker='o' if sec in sectors[:3] else None, markersize=3, label=f"Sector {sec}")

    titles = {
        "price": "Price", "output": "Output", "export": "Export Value",
        "consumption_domestic": "Domestic Consumption", "consumption_import": "Import Consumption",
        "intermediate_domestic": "Domestic Intermediate Supply", "intermediate_import": "Import Intermediate Supply",
    }
    title = f"{country} - {titles.get(metric, metric)}{ylabel_suffix}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Periods", fontsize=10)
    ax.set_ylabel(titles.get(metric, metric) + ylabel_suffix, fontsize=10)
    if relative:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_diagnostics(sim, save_path: Optional[str] = None, show: bool = False, warmup: int = 0) -> None:
    summary = sim.summarize_history()
    total_len = len(sim.history["H"])
    trim = max(int(warmup or 0), 0)
    if total_len > 0 and trim >= total_len:
        trim = max(total_len - 1, 0)
    periods = np.arange(total_len - trim)
    data = _trimmed_summary(summary, trim)
    data_H = data["H"]
    data_F = data["F"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.plot(periods, data_H["income"], 'b-', linewidth=2, label="Home", marker='o', markersize=2)
    ax1.plot(periods, data_F["income"], 'r--', linewidth=2, label="Foreign", marker='s', markersize=2)
    ax1.set_title("Income Levels", fontweight='bold'); ax1.set_ylabel("Income"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(periods, data_H["trade_balance"], 'g-', linewidth=2, label="Home Trade Balance")
    ax2.plot(periods, data_F["trade_balance"], 'm--', linewidth=2, label="Foreign Trade Balance")
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title("Trade Balance", fontweight='bold'); ax2.set_ylabel("Trade Balance"); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.plot(periods, data_H["income_growth"], 'b-', linewidth=2, label="Home")
    ax3.plot(periods, data_F["income_growth"], 'r--', linewidth=2, label="Foreign")
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title("Income Growth Rate", fontweight='bold'); ax3.set_ylabel("Growth Rate (%)"); ax3.set_xlabel("Periods"); ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4.axis('off')
    h_final_growth = data_H["income_growth"][-1]; f_final_growth = data_F["income_growth"][-1]
    h_vol = np.std(data_H["income_growth"]); f_vol = np.std(data_F["income_growth"])
    stats_text = (
        f"Model Diagnostic Summary ({len(periods)} periods)\n\n"
        f"Income Performance:\n• Home Final Growth Rate: {h_final_growth:.1f}%\n• Foreign Final Growth Rate: {f_final_growth:.1f}%\n\n"
        f"Volatility Analysis:\n• Home Income Volatility (σ): {h_vol:.1f}%\n• Foreign Income Volatility (σ): {f_vol:.1f}%\n\n"
        f"Trade Status:\n• Home Final Trade Balance: {data_H['trade_balance'][-1]:.2f}\n"
        f"• Foreign Final Trade Balance: {data_F['trade_balance'][-1]:.2f}\n"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3), va='top')

    plt.suptitle("Two-Country Economic Simulation - Dashboard", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_history_agent_view(
    sim,
    agent_log: List[dict],
    k_per_step: int,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    annotate_decisions: bool = True,
    warmup: Optional[int] = None,
) -> None:
    """智能体视角的历史图（裁剪 warmup，并标注每回合决策）。

    参数：
    - sim: 双国仿真器
    - agent_log: 智能体循环日志（每项含 {t, obs, action, reward}）
    - k_per_step: 每回合推进的步数（用于定位回合边界）
    - metrics: 指标列表（默认 [income, output_sum, trade_balance, income_growth]）
    - save_path/show: 与常规绘图一致
    - annotate_decisions: 是否在每个回合起点处标注决策摘要
    - warmup: 若提供则显式裁剪前 warmup 期，否则依据 rounds*k 自动推断

    最小可行约定：
    - 仅支持两国 'H' 与 'F' 的汇总曲线；
    - 回合标注在“回合起点”（即采取决策之前的状态位置）处画竖线与简短文本摘要；
    - X 轴以智能体开始博弈时刻为 0 重新计数。
    """
    metrics = metrics or ["income", "output_sum", "trade_balance", "income_growth"]
    summary = sim.summarize_history()

    total_len = int(len(sim.history.get("H", [])))
    rounds = int(len(agent_log or []))
    k = int(max(k_per_step, 1))
    warmup_auto = max(total_len - 1 - rounds * k, 0)
    warmup_steps = warmup_auto if warmup is None else max(int(warmup), 0)
    if total_len > 0 and warmup_steps >= total_len:
        warmup_steps = max(total_len - 1, 0)

    # 重新索引：从 warmup 结束时刻开始
    periods = np.arange(total_len - warmup_steps)
    data = _trimmed_summary(summary, warmup_steps)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = {'H': '#1f77b4', 'F': '#ff7f0e'}
    linestyles = {'H': '-', 'F': '--'}

    for ax, metric in zip(axes, metrics):
        if metric not in data["H"]:
            continue
        series_H = np.array(data["H"][metric])
        series_F = np.array(data["F"][metric])

        ax.plot(periods, series_H, color=colors['H'], linestyle=linestyles['H'],
                linewidth=2, marker='o', markersize=2, label="Home")
        ax.plot(periods, series_F, color=colors['F'], linestyle=linestyles['F'],
                linewidth=2, marker='s', markersize=2, label="Foreign")
        ylabel = metric.replace("_", " ").title()
        if "growth" in metric:
            ylabel += " (%)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        if "growth" in metric:
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel("Agent Rounds (steps reindexed)", fontsize=10)
    fig.suptitle("Two-Country Simulation — Agent View (trim warmup)", fontsize=14, fontweight='bold')

    # 在第一幅子图上标记回合起点与决策摘要
    if annotate_decisions and rounds > 0:
        ax0 = axes[0]
        # 回合起点（裁剪后相对坐标）：0, k, 2k, ...
        round_x = [i * k for i in range(rounds)]
        # 竖线
        for x in round_x:
            ax0.axvline(x=x, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

        # 文本摘要（尽量简短，KISS）
        def _short_summary(action_map: dict) -> str:
            if not isinstance(action_map, dict):
                return ""
            def _cnt(d: dict | None) -> int:
                return len(d or {}) if isinstance(d, dict) else 0
            H = action_map.get('H', {}) or {}
            F = action_map.get('F', {}) or {}
            h_s = f"H τ:{_cnt(H.get('import_tariff'))} q:{_cnt(H.get('export_quota'))}"
            f_s = f"F τ:{_cnt(F.get('import_tariff'))} q:{_cnt(F.get('export_quota'))}"
            return f"{h_s} | {f_s}"

        ymax = max(np.max(np.array(data['H'][metrics[0]])),
                   np.max(np.array(data['F'][metrics[0]])))
        y_text = ymax
        for i, x in enumerate(round_x):
            # 标注略微错位，避免重叠
            dy = (0.04 * ymax) * (1 if (i % 2 == 0) else 0.6)
            item = agent_log[i] if i < len(agent_log) else {}
            label = _short_summary(item.get('action', {}))
            if not label:
                continue
            ax0.annotate(label, xy=(x, y_text), xytext=(0, dy), textcoords='offset points',
                         fontsize=7, rotation=90, va='bottom', ha='center', color='dimgray',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, linewidth=0.0))

    plt.tight_layout(); plt.subplots_adjust(top=0.90, bottom=0.08)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
