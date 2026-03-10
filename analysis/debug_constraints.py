"""深入分析均衡求解器的约束结构和缺失的约束。"""

import sys
sys.path.insert(0, "/mnt/d/eco-model")

import numpy as np
import torch
from analysis.model import (
    create_symmetric_parameters,
    normalize_model_params,
    solve_initial_equilibrium,
    compute_output,
    compute_marginal_cost,
    armington_share,
)
from analysis.model.model import EquilibriumLayout, _country_block, EPS, TORCH_DTYPE

def analyze_equilibrium_constraints():
    """分析均衡求解器的约束结构。"""
    print("=" * 70)
    print("均衡求解器约束结构分析")
    print("=" * 70)
    
    params_raw = create_symmetric_parameters()
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params_raw)
    
    n = params.home.alpha.shape[0]
    layout = EquilibriumLayout(n, params.tradable_idx)
    
    # 提取解状态
    state_H = {
        "price": torch.tensor(eqm["H"]["prices"]["P_j"], dtype=TORCH_DTYPE),
        "X_dom": torch.tensor(eqm["H"]["intermediate_inputs"]["X_ij"], dtype=TORCH_DTYPE),
        "X_imp": torch.tensor(eqm["H"]["intermediate_inputs"]["X_O_ij"][:, params.tradable_idx], dtype=TORCH_DTYPE),
        "C_dom": torch.tensor(eqm["H"]["final_consumption"]["C_j"], dtype=TORCH_DTYPE) + 
                 torch.tensor(eqm["H"]["final_consumption"]["C_I_j"], dtype=TORCH_DTYPE),
        "C_imp": torch.tensor(eqm["H"]["final_consumption"]["C_O_j"][params.tradable_idx], dtype=TORCH_DTYPE),
    }
    state_F = {
        "price": torch.tensor(eqm["F"]["prices"]["P_j"], dtype=TORCH_DTYPE),
        "X_dom": torch.tensor(eqm["F"]["intermediate_inputs"]["X_ij"], dtype=TORCH_DTYPE),
        "X_imp": torch.tensor(eqm["F"]["intermediate_inputs"]["X_O_ij"][:, params.tradable_idx], dtype=TORCH_DTYPE),
        "C_dom": torch.tensor(eqm["F"]["final_consumption"]["C_j"], dtype=TORCH_DTYPE) + 
                 torch.tensor(eqm["F"]["final_consumption"]["C_I_j"], dtype=TORCH_DTYPE),
        "C_imp": torch.tensor(eqm["F"]["final_consumption"]["C_O_j"][params.tradable_idx], dtype=TORCH_DTYPE),
    }
    
    print("\n【当前均衡求解器包含的约束】")
    print("-" * 50)
    print("1. 零利润条件: P_i = λ_i (价格 = 边际成本)")
    print("2. 中间品需求条件: X_{ij} = α_{ij} * θ * P_i * Y_i / P_j")
    print("3. 消费需求条件: C_j = β_j * θ_c * I / P_j")
    print("4. 贸易收支平衡: 出口价值 = 进口价值")
    print("5. 价格锚定: P_H[0] = P_F[0] = 1")
    
    print("\n【缺失的约束 - 供需平衡】")
    print("-" * 50)
    print("❌ 商品市场出清: Y_j = Σ_i X_{ij} + C_j + Export_j")
    print("   (当前求解器没有显式包含这个约束!)")
    
    # 验证各个约束的满足程度
    print("\n" + "=" * 70)
    print("约束满足程度验证")
    print("=" * 70)
    
    # 1. 零利润条件
    print("\n【约束1】零利润条件:")
    X_dom_full = torch.tensor(eqm["H"]["intermediate_inputs"]["X_ij"], dtype=TORCH_DTYPE)
    X_imp_full = torch.tensor(eqm["H"]["intermediate_inputs"]["X_O_ij"], dtype=TORCH_DTYPE)
    import_prices_H = params.home.import_cost * state_F["price"]
    
    outputs_H = compute_output(params.home, X_dom_full, X_imp_full, layout.tradable_mask)
    lambdas_H = compute_marginal_cost(params.home, state_H["price"], import_prices_H, layout.tradable_mask)
    
    print(f"  Prices:       {state_H['price'].numpy()}")
    print(f"  Marginal Cost:{lambdas_H.numpy()}")
    print(f"  差值:          {(state_H['price'] - lambdas_H).numpy()}")
    print(f"  最大误差:      {torch.abs(state_H['price'] - lambdas_H).max().item():.6e}")
    
    # 2. 供需平衡（缺失的约束）
    print("\n【缺失约束】供需平衡 (商品市场出清):")
    
    # H国的对方进口需求 = H国的出口
    X_imp_from_F = torch.tensor(eqm["F"]["intermediate_inputs"]["X_O_ij"], dtype=TORCH_DTYPE)
    C_imp_from_F = torch.tensor(eqm["F"]["final_consumption"]["C_O_j"], dtype=TORCH_DTYPE)
    export_H = X_imp_from_F.sum(dim=0) + C_imp_from_F  # F从H进口 = H的出口
    
    # 总需求 = 国内中间品使用 + 国内消费 + 出口
    demand_H = X_dom_full.sum(dim=0) + state_H["C_dom"] + export_H
    
    print(f"  H国 Supply (Y): {outputs_H.numpy()}")
    print(f"  H国 Demand:     {demand_H.numpy()}")
    print(f"  H国 Gap (D-S):  {(demand_H - outputs_H).numpy()}")
    print(f"  最大 Gap:       {torch.abs(demand_H - outputs_H).max().item():.6e}")
    
    # 3. 贸易收支
    print("\n【约束4】贸易收支平衡:")
    export_value_H = (state_H["price"] * export_H).sum()
    C_imp_full = torch.tensor(eqm["H"]["final_consumption"]["C_O_j"], dtype=TORCH_DTYPE)
    import_value_H = (import_prices_H * (X_imp_full.sum(dim=0) + C_imp_full)).sum()
    print(f"  H国出口价值: {export_value_H.item():.6f}")
    print(f"  H国进口价值: {import_value_H.item():.6f}")
    print(f"  差额:        {(export_value_H - import_value_H).item():.6e}")
    
    return params, eqm, layout

def propose_solution():
    """提出解决方案。"""
    print("\n" + "=" * 70)
    print("解决方案分析")
    print("=" * 70)
    
    print("""
【问题根源】
当前均衡求解器使用"需求驱动"的方法：
1. 给定价格 → 计算需求（基于一阶条件）
2. 给定需求 → 计算产出（生产函数）
3. 但没有约束：产出 = 总需求

这意味着解出的"产出"仅仅是"给定投入的产出"，而非"满足市场出清的产出"。

【方案分析】

方案 A: 添加供需平衡残差
  - 在 _country_block 中添加 n 个新残差：Y_j - (Σ_i X_{ij} + C_j + Export_j) = 0
  - 优点：直接解决问题
  - 缺点：增加残差数量，可能影响收敛性；需要仔细检查自由度匹配

方案 B: 重新设计变量与约束
  - 当前变量：X_dom, X_imp, C_dom, C_imp, price (共 2*n*n + 2*n*k + 2*n + 2*n + 2*n)
  - 考虑添加 Y（产出）作为独立变量
  - 需求由一阶条件内生决定
  - 约束：零利润 + 市场出清 + 贸易平衡

方案 C: 迭代校正
  - 求解当前均衡后，运行若干期热启动让系统收敛
  - 当 Gap 足够小时，使用该状态作为新的起点
  - 优点：不修改核心求解器
  - 缺点：不是真正的均衡解

方案 D: 使用数值优化约束
  - 将供需平衡作为硬约束（而非软残差）
  - 使用 scipy.optimize.minimize with constraints
  - 目标函数：其他残差的平方和
  - 约束：供需平衡恒等式

【推荐方案】
方案 A（添加供需平衡残差）最直接且改动最小。
    """)

if __name__ == "__main__":
    analyze_equilibrium_constraints()
    propose_solution()
