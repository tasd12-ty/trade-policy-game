"""博弈结果可视化模块。

参考 grad_op/analysis/optimization/plotting.py 实现。
支持热启动压缩显示、政策事件标注、多指标面板。

依赖：matplotlib, numpy
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置 ====================
WARMUP_WIDTH_RATIO = 0.1  # 热启动在图表中占的宽度比例


# ==================== 数据提取 ====================

def summarize_history(
    sim,
    base_period: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """从 TwoCountrySimulator.history 提取时间序列 summary。

    返回：
        {
            "H": {"income": array, "trade_balance_val": array,
                  "price_mean": array, "output_growth": array},
            "F": { ... }
        }
    """
    summary: Dict[str, Dict[str, np.ndarray]] = {}

    for c in ["H", "F"]:
        hist = sim.history[c]
        cp = sim.params.home if c == "H" else sim.params.foreign
        Nl = cp.Nl

        n = len(hist)
        income = np.array([s.income for s in hist])
        price_mean = np.array([s.price[:Nl].mean() for s in hist])
        output_sum = np.array([s.output[:Nl].sum() for s in hist])

        # 贸易差额：export_value - import_value
        tb = np.zeros(n)
        for t, s in enumerate(hist):
            exp_val = float(np.dot(s.price[:Nl], s.export_actual[:Nl]))
            imp_val = float(np.dot(s.imp_price, s.X_imp.sum(axis=0) + s.C_imp))
            tb[t] = exp_val - imp_val

        # 产出增长率 (%)
        base_output = max(output_sum[base_period], 1e-9)
        output_growth = (output_sum / base_output - 1.0) * 100.0

        summary[c] = {
            "income": income,
            "trade_balance_val": tb,
            "price_mean": price_mean,
            "output_growth": output_growth,
        }

    return summary


# ==================== 坐标压缩 ====================

def _compress_x_axis(x, warmup_end, total_steps, width_ratio=WARMUP_WIDTH_RATIO):
    """对 x 轴进行非线性映射，压缩热启动阶段。"""
    x = np.array(x)
    x_compressed = np.zeros_like(x, dtype=float)

    warmup_mask = x <= warmup_end
    game_mask = x > warmup_end

    if warmup_end > 0:
        x_compressed[warmup_mask] = (x[warmup_mask] / warmup_end) * width_ratio

    game_range = total_steps - warmup_end
    if game_range > 0:
        x_compressed[game_mask] = width_ratio + (
            (x[game_mask] - warmup_end) / game_range
        ) * (1 - width_ratio)

    return x_compressed


def _create_compressed_ticks(warmup_end, total_steps, events=None,
                             width_ratio=WARMUP_WIDTH_RATIO):
    """创建压缩后的刻度位置和标签。"""
    warmup_labels = [0, warmup_end]
    warmup_positions = [0, width_ratio]

    if events and len(events) > 0:
        event_periods = sorted(set(
            e.get("period", 0) for e in events
            if e.get("period", 0) > warmup_end
        ))
        game_labels = event_periods
    else:
        game_steps = total_steps - warmup_end
        if game_steps <= 10:
            step = 1
        elif game_steps <= 50:
            step = 5
        else:
            step = game_steps // 5
        game_labels = list(range(warmup_end + step, total_steps + 1, step))

    if total_steps not in game_labels:
        game_labels.append(total_steps)
    game_labels = [la for la in game_labels if la > warmup_end]

    game_range = total_steps - warmup_end
    if game_range > 0:
        game_positions = [
            width_ratio + ((la - warmup_end) / game_range) * (1 - width_ratio)
            for la in game_labels
        ]
    else:
        game_positions = []

    return warmup_positions + game_positions, warmup_labels + game_labels


def _add_warmup_shading(ax, width_ratio):
    """在热启动区域添加灰色背景。"""
    ax.axvspan(0, width_ratio, alpha=0.15, color='gray', label='Warmup')
    ax.axvline(width_ratio, color='gray', linestyle='--', alpha=0.5, linewidth=1)


# ==================== 事件聚合 ====================

def _format_sector_mapping(mapping):
    """将 {sector: value} 格式化为短文本。"""
    if not mapping:
        return "{}"
    try:
        keys = sorted(mapping.keys(), key=lambda k: int(k))
    except Exception:
        keys = sorted(mapping.keys(), key=str)
    parts = []
    for k in keys:
        try:
            kk = int(k)
        except Exception:
            kk = k
        parts.append(f"{kk}:{mapping[k]:.2f}")
    return "{" + ", ".join(parts) + "}"


def _aggregate_decisions_from_events(events):
    """将 policy_events 聚合成 {period: {'H':{'tariff':{},'quota':{}}, 'F':{...}}}。"""
    out = {}
    if not events:
        return out
    for e in events:
        if not isinstance(e, dict):
            continue
        period = e.get("period")
        if period is None:
            continue
        t = int(period)
        country = (e.get("country") or "").upper()
        if country not in ("H", "F"):
            continue
        ev_type = e.get("type")
        sectors = e.get("sectors") or {}
        if not isinstance(sectors, dict):
            continue
        bucket = out.setdefault(t, {}).setdefault(country, {})
        if ev_type == "import_tariff":
            bucket.setdefault("tariff", {}).update(sectors)
        elif ev_type == "export_control":
            bucket.setdefault("quota", {}).update(sectors)
    return out


# ==================== 主图表 ====================

def plot_game_analysis(
    summary: Dict,
    events: Optional[List[Dict]],
    save_path: str,
    warmup_periods: int = 0,
):
    """绘制博弈分析图表（收入、贸易差额、价格、产出增长 + 决策时间线）。"""
    metrics = ["income", "trade_balance_val", "price_mean", "output_growth"]
    metric_labels = {
        "income": "Income",
        "trade_balance_val": "Trade Balance",
        "price_mean": "Price Index (Mean)",
        "output_growth": "Real GDP Growth (%)",
    }

    steps = len(summary["H"]["income"])
    x_orig = np.arange(steps)

    use_compression = warmup_periods > 0 and warmup_periods < steps - 5

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.7])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_timeline = fig.add_subplot(gs[2, :])
    decisions_by_t = _aggregate_decisions_from_events(events)

    for i, key in enumerate(metrics):
        ax = axes[i]
        if key not in summary["H"]:
            continue
        val_H = np.array(summary["H"][key])
        val_F = np.array(summary["F"][key])

        if use_compression:
            x_plot = _compress_x_axis(x_orig, warmup_periods, steps - 1)
            ax.plot(x_plot, val_H, label="H", color="blue", linewidth=1.5)
            ax.plot(x_plot, val_F, label="F", color="red", linewidth=1.5)

            tick_pos, tick_labels = _create_compressed_ticks(
                warmup_periods, steps - 1, events=events,
            )
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlim(0, 1)
            _add_warmup_shading(ax, WARMUP_WIDTH_RATIO)

            for e in (events or []):
                t = e.get("period", 0)
                if t < steps:
                    tc = _compress_x_axis([t], warmup_periods, steps - 1)[0]
                    ax.axvline(tc, color="green", linestyle="--", alpha=0.7, linewidth=1)
        else:
            ax.plot(x_orig, val_H, label="H", color="blue")
            ax.plot(x_orig, val_F, label="F", color="red")
            for e in (events or []):
                t = e.get("period", 0)
                if t < steps:
                    ax.axvline(t, color="gray", linestyle="--", alpha=0.5)

        ax.set_title(metric_labels.get(key, key))
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle=":", alpha=0.6)

    # 决策时间线
    ax_timeline.set_title("Policy Decisions Timeline", fontsize=10)
    ax_timeline.set_yticks([1, 0])
    ax_timeline.set_yticklabels(["H", "F"])
    ax_timeline.set_ylim(-0.7, 1.7)
    ax_timeline.grid(True, axis="y", linestyle=":", alpha=0.4)

    if use_compression:
        tick_pos, tick_labels = _create_compressed_ticks(
            warmup_periods, steps - 1, events=events,
        )
        ax_timeline.set_xticks(tick_pos)
        ax_timeline.set_xticklabels(tick_labels, fontsize=8)
        ax_timeline.set_xlim(0, 1)
        _add_warmup_shading(ax_timeline, WARMUP_WIDTH_RATIO)

        def x_map(t_val):
            return float(_compress_x_axis([t_val], warmup_periods, steps - 1)[0])
    else:
        ax_timeline.set_xlim(0, steps - 1)

        def x_map(t_val):
            return float(t_val)

    for t in sorted(decisions_by_t.keys()):
        if t < 0 or t >= steps:
            continue
        x = x_map(int(t))
        ax_timeline.axvline(x, color="green", linestyle="--", alpha=0.35, linewidth=1)

        for y, c, color in [(1, "H", "blue"), (0, "F", "red")]:
            d = decisions_by_t.get(int(t), {}).get(c, {})
            tar = _format_sector_mapping(d.get("tariff", {}))
            quo = _format_sector_mapping(d.get("quota", {}))
            text = f"t{tar}\nq{quo}"
            ax_timeline.text(
                x, y, text,
                ha="center", va="center", fontsize=7, color=color,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "alpha": 0.75, "edgecolor": "none"},
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
