"""分析冷启动均衡与热启动动态的供需平衡状况。"""

import sys
sys.path.insert(0, "/mnt/d/eco-model")

import numpy as np
import torch
from analysis.model import (
    create_symmetric_parameters,
    normalize_model_params,
    solve_initial_equilibrium,
)
from analysis.model.sim import TwoCountryDynamicSimulator, _extract_export_base, _build_state

def analyze_equilibrium():
    """分析冷启动均衡的供需状况。"""
    print("=" * 60)
    print("分析冷启动（静态均衡）的供需平衡状况")
    print("=" * 60)
    
    params_raw = create_symmetric_parameters()
    params = normalize_model_params(params_raw)
    eqm = solve_initial_equilibrium(params_raw)
    
    print(f"\n收敛信息: {eqm['convergence_info']}")
    
    # 计算冷启动时的供需
    tmask = np.zeros(params.home.alpha.shape[0], bool)
    tmask[params.tradable_idx] = True
    
    export_H = _extract_export_base(eqm["F"])
    export_F = _extract_export_base(eqm["H"])
    
    state_H = _build_state(eqm["H"], params.home, export_H, tmask)
    state_F = _build_state(eqm["F"], params.foreign, export_F, tmask)
    
    print("\n【H国】冷启动供需分析:")
    supply_H = state_H.output.detach().numpy()
    # 需求 = 中间品使用 + 消费 + 出口
    demand_H = (state_H.X_dom.sum(dim=0) + state_H.C_dom + state_H.export_base).detach().numpy()
    gap_H = demand_H - supply_H
    
    print(f"  Supply:  {supply_H}")
    print(f"  Demand:  {demand_H}")
    print(f"  Gap:     {gap_H}")
    print(f"  Max Gap: {np.abs(gap_H).max():.6e}")
    
    print("\n【F国】冷启动供需分析:")
    supply_F = state_F.output.detach().numpy()
    demand_F = (state_F.X_dom.sum(dim=0) + state_F.C_dom + state_F.export_base).detach().numpy()
    gap_F = demand_F - supply_F
    
    print(f"  Supply:  {supply_F}")
    print(f"  Demand:  {demand_F}")
    print(f"  Gap:     {gap_F}")
    print(f"  Max Gap: {np.abs(gap_F).max():.6e}")
    
    return params, eqm

def analyze_warmup(params, eqm, warmup_periods=10):
    """分析热启动期间的供需演变。"""
    print("\n" + "=" * 60)
    print(f"分析热启动（动态仿真 {warmup_periods} 期）的供需平衡状况")
    print("=" * 60)
    
    sim = TwoCountryDynamicSimulator(params, eqm, theta_price=0.05)
    
    # 收集初始状态
    gaps_H = []
    gaps_F = []
    
    for t in range(warmup_periods + 1):
        if t > 0:
            sim.step()
        
        state_H = sim.home_state
        state_F = sim.foreign_state
        
        supply_H = state_H.output.detach().cpu().numpy()
        demand_H = (state_H.X_dom.sum(dim=0) + state_H.C_dom + state_H.export_base).detach().cpu().numpy()
        gap_H = demand_H - supply_H
        
        supply_F = state_F.output.detach().cpu().numpy()
        demand_F = (state_F.X_dom.sum(dim=0) + state_F.C_dom + state_F.export_base).detach().cpu().numpy()
        gap_F = demand_F - supply_F
        
        gaps_H.append(gap_H)
        gaps_F.append(gap_F)
        
        if t <= 3 or t == warmup_periods:
            print(f"\n期 {t}:")
            print(f"  H Gap: {gap_H}, Max: {np.abs(gap_H).max():.6e}")
            print(f"  F Gap: {gap_F}, Max: {np.abs(gap_F).max():.6e}")
            print(f"  H export_base: {state_H.export_base.detach().cpu().numpy()}")
            print(f"  H export_actual: {state_H.export_actual.detach().cpu().numpy()}")

    print("\n" + "-" * 40)
    print("结论:")
    print(f"  初始 Max Gap (H): {np.abs(gaps_H[0]).max():.6e}")
    print(f"  最终 Max Gap (H): {np.abs(gaps_H[-1]).max():.6e}")
    print(f"  初始 Max Gap (F): {np.abs(gaps_F[0]).max():.6e}")
    print(f"  最终 Max Gap (F): {np.abs(gaps_F[-1]).max():.6e}")

if __name__ == "__main__":
    params, eqm = analyze_equilibrium()
    analyze_warmup(params, eqm)
