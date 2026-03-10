"""
LLM vs Gradient 博弈结果对比绘图工具（修正版）。

根据 JSON 实验结果文件生成对比图表：
- 根据 config.llm.llm_plays 确定哪个国家使用 LLM
- LLM 方法使用虚线 (--)
- 梯度方法使用实线 (-)

用法：
    python -m analysis.optimization.compare_results \
        --auto results_0106_llmH_strategyF_F/llm_vs_gradient_policies.json \
        --output test.png
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_result(json_path: str) -> dict:
    """加载 JSON 实验结果文件。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_method_mapping(result: dict) -> Dict[str, str]:
    """
    从实验配置中确定 H/F 分别对应哪种方法。
    
    Returns:
        {"H": "llm" or "gradient", "F": "llm" or "gradient"}
    """
    config = result.get("config", {})
    llm_config = config.get("llm", {})
    
    # 检查是否有 llm 配置
    if not llm_config:
        # 没有 llm 配置，这是纯梯度实验
        return {"H": "gradient", "F": "gradient"}
    
    llm_plays = llm_config.get("llm_plays", "both")
    
    if llm_plays == "H":
        return {"H": "llm", "F": "gradient"}
    elif llm_plays == "F":
        return {"H": "gradient", "F": "llm"}
    elif llm_plays == "both":
        return {"H": "llm", "F": "llm"}
    else:
        # 尝试从 reasoning 字段推断
        policies = result.get("policies", [])
        if policies:
            reasoning = policies[0].get("reasoning", {})
            h_reason = reasoning.get("H", "")
            f_reason = reasoning.get("F", "")
            h_method = "llm" if "[LLM]" in str(h_reason) else "gradient"
            f_method = "llm" if "[LLM]" in str(f_reason) else "gradient"
            return {"H": h_method, "F": f_method}
        return {"H": "gradient", "F": "gradient"}


def extract_metrics_by_method(result: dict) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    从实验结果中提取每回合的指标，按方法类型分组。
    
    Returns:
        {"llm": {round: {metric: value}}, "gradient": {round: {metric: value}}}
    """
    policies = result.get("policies", [])
    method_mapping = get_method_mapping(result)
    
    metrics_by_method = {"llm": {}, "gradient": {}}
    
    for p in policies:
        round_num = p.get("round", 0)
        payoff = p.get("payoff", {})
        metrics = p.get("metrics", {})
        
        # 为每种方法提取对应国家的指标
        for country, method in method_mapping.items():
            country_payoff = payoff.get(country, 0)
            country_income = metrics.get(f"income_{country}", 0)
            country_tb = metrics.get(f"trade_balance_{country}", 0)
            country_price = metrics.get(f"price_mean_{country}", 0)
            
            if round_num not in metrics_by_method[method]:
                metrics_by_method[method][round_num] = {
                    "payoff": country_payoff,
                    "income": country_income,
                    "trade_balance": country_tb,
                    "price_mean": country_price,
                }
            else:
                # 如果两国用同一方法（如 both=llm），取平均
                existing = metrics_by_method[method][round_num]
                existing["payoff"] = (existing["payoff"] + country_payoff) / 2
                existing["income"] = (existing["income"] + country_income) / 2
                existing["trade_balance"] = (existing["trade_balance"] + country_tb) / 2
                existing["price_mean"] = (existing["price_mean"] + country_price) / 2
    
    return metrics_by_method


def extract_policy_by_method(result: dict) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    """
    从实验结果中提取每回合的政策，按方法类型分组。
    
    Returns:
        {"llm": {round: {tariff: {sector: value}, quota: {...}}}, 
         "gradient": {...}}
    """
    policies = result.get("policies", [])
    method_mapping = get_method_mapping(result)
    
    policy_by_method = {"llm": {}, "gradient": {}}
    
    for p in policies:
        round_num = p.get("round", 0)
        decision = p.get("decision", {})
        
        for country, method in method_mapping.items():
            country_decision = decision.get(country, {})
            tariff = country_decision.get("tariff", {})
            quota = country_decision.get("quota", {})
            
            # 转换 key 为 str 以保持一致性
            tariff_clean = {str(k): float(v) for k, v in tariff.items()}
            quota_clean = {str(k): float(v) for k, v in quota.items()}
            
            if round_num not in policy_by_method[method]:
                policy_by_method[method][round_num] = {
                    "tariff": tariff_clean,
                    "quota": quota_clean,
                }
            else:
                # 如果两国用同一方法，合并/平均
                existing = policy_by_method[method][round_num]
                for k, v in tariff_clean.items():
                    if k in existing["tariff"]:
                        existing["tariff"][k] = (existing["tariff"][k] + v) / 2
                    else:
                        existing["tariff"][k] = v
                for k, v in quota_clean.items():
                    if k in existing["quota"]:
                        existing["quota"][k] = (existing["quota"][k] + v) / 2
                    else:
                        existing["quota"][k] = v
    
    return policy_by_method


def plot_single_experiment(
    result: dict,
    save_path: str,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    绑制单个实验的 LLM vs Gradient 对比图。
    """
    method_mapping = get_method_mapping(result)
    metrics_by_method = extract_metrics_by_method(result)
    policy_by_method = extract_policy_by_method(result)
    
    # 检查是否有两种不同的方法
    has_both = (method_mapping["H"] != method_mapping["F"])
    
    if not has_both:
        # 如果只有一种方法，使用简化的图表
        single_method = method_mapping["H"]
        print(f"Warning: Both countries use {single_method}, showing single-method plot")
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # 线型和颜色配置
    style_config = {
        "llm": {"linestyle": "--", "marker": "o", "color": "#1f77b4", "label": "LLM"},
        "gradient": {"linestyle": "-", "marker": "s", "color": "#ff7f0e", "label": "Strategy (Gradient)"},
    }
    
    subplot_configs = [
        ("tariff_mean", "Average Tariff", "Tariff Rate"),
        ("payoff", "Payoff", "Payoff"),
        ("income", "Income", "Income"),
        ("trade_balance", "Trade Balance", "Trade Balance"),
        ("price_mean", "Price Index", "Price"),
        ("quota_mean", "Average Quota", "Quota"),
    ]
    
    for idx, (key, title, ylabel) in enumerate(subplot_configs):
        ax = axes[idx]
        
        for method in ["llm", "gradient"]:
            metrics = metrics_by_method.get(method, {})
            policy = policy_by_method.get(method, {})
            
            if not metrics and not policy:
                continue
            
            rounds = sorted(metrics.keys()) if metrics else sorted(policy.keys())
            
            if not rounds:
                continue
            
            if key == "tariff_mean":
                values = []
                for r in rounds:
                    tariffs = policy.get(r, {}).get("tariff", {})
                    if tariffs:
                        values.append(np.mean(list(tariffs.values())))
                    else:
                        values.append(0)
            elif key == "quota_mean":
                values = []
                for r in rounds:
                    quotas = policy.get(r, {}).get("quota", {})
                    if quotas:
                        values.append(np.mean(list(quotas.values())))
                    else:
                        values.append(1.0)
            else:
                values = [metrics.get(r, {}).get(key, 0) for r in rounds]
            
            if not values:
                continue
            
            style = style_config[method]
            ax.plot(
                rounds, values,
                label=style["label"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5,
                color=style["color"],
                linewidth=1.5,
            )
        
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # 获取实验名称和生成合适的标题
    config = result.get("config", {})
    exp_name = config.get("name", "Experiment")
    llm_config = config.get("llm", {})
    
    if not llm_config:
        subtitle = "Pure Gradient"
    else:
        llm_plays = llm_config.get("llm_plays", "both")
        if llm_plays == "both":
            subtitle = "Both use LLM"
        else:
            subtitle = f"LLM plays: {llm_plays}"
    
    plt.suptitle(f"LLM vs Strategy Comparison\n{exp_name} ({subtitle})", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_tariff_by_sector_single(
    result: dict,
    save_path: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    绑制单个实验中每个部门关税的 LLM vs Gradient 对比。
    """
    policy_by_method = extract_policy_by_method(result)
    
    # 收集所有部门
    all_sectors = set()
    for method in ["llm", "gradient"]:
        policy = policy_by_method.get(method, {})
        for round_data in policy.values():
            all_sectors.update(round_data.get("tariff", {}).keys())
    
    sectors = sorted(all_sectors, key=lambda x: int(x) if str(x).isdigit() else 999)
    n_sectors = len(sectors)
    
    if n_sectors == 0:
        print("No sector data found")
        return
    
    cols = min(3, n_sectors)
    rows = (n_sectors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_sectors == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    style_config = {
        "llm": {"linestyle": "--", "marker": "o", "color": "#1f77b4", "label": "LLM"},
        "gradient": {"linestyle": "-", "marker": "s", "color": "#ff7f0e", "label": "Strategy"},
    }
    
    for s_idx, sector in enumerate(sectors):
        ax = axes[s_idx]
        
        for method in ["llm", "gradient"]:
            policy = policy_by_method.get(method, {})
            rounds = sorted(policy.keys())
            
            if not rounds:
                continue
            
            values = []
            for r in rounds:
                tariff = policy.get(r, {}).get("tariff", {})
                val = tariff.get(sector, tariff.get(str(sector), 0))
                values.append(float(val))
            
            if not values:
                continue
            
            style = style_config[method]
            ax.plot(
                rounds, values,
                label=style["label"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                color=style["color"],
                linewidth=1.5,
            )
        
        ax.set_title(f"Sector {sector}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Tariff")
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    for k in range(n_sectors, len(axes)):
        axes[k].axis("off")
    
    config = result.get("config", {})
    exp_name = config.get("name", "Experiment")
    
    plt.suptitle(f"Tariff by Sector: {exp_name}", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sector plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM vs Gradient game results with plotting"
    )
    parser.add_argument(
        "--auto", "-a",
        nargs="+",
        required=True,
        help="Input JSON file(s)"
    )
    parser.add_argument(
        "--output", "-o",
        default="comparison.png",
        help="Output image path (default: comparison.png)"
    )
    parser.add_argument(
        "--sector-plot",
        action="store_true",
        help="Also generate per-sector tariff plot"
    )
    
    args = parser.parse_args()
    
    for path in args.auto:
        try:
            result = load_experiment_result(path)
            method_mapping = get_method_mapping(result)
            print(f"Loaded: {path}")
            print(f"  Method mapping: H={method_mapping['H']}, F={method_mapping['F']}")
            
            # 生成输出路径
            if args.output:
                output_path = args.output
            else:
                output_path = str(Path(path).parent / "test.png")
            
            # 生成主图
            plot_single_experiment(result, output_path)
            
            # 可选：生成部门图
            if args.sector_plot:
                sector_path = str(Path(output_path).with_suffix("")) + "_sectors.png"
                plot_tariff_by_sector_single(result, sector_path)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
