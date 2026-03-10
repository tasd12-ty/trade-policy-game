"""敏感性调试工具

用于调试和验证仿真器对政策变化的敏感性。

测试场景：
1. 基线场景 - 无政策干预
2. 高关税冲击 - 对可贸易部门征收 20% 关税

验证内容：
- 收入变化是否显著
- 价格变化是否显著
- 目标函数对政策变化的响应

用途：
- 验证模型的敏感性
- 调试优化器效果不佳的原因
- 检查目标函数是否正确计算
"""

import sys
from pathlib import Path
import numpy as np
import logging

# 兼容直接执行：将仓库根目录加入 sys.path，确保可导入 analysis 命名空间
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.model import bootstrap_simulator, create_symmetric_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sensitivity():
    # 1. Baseline
    params = create_symmetric_parameters()
    sim_base = bootstrap_simulator(params)
    sim_base.run(20)
    base_res = sim_base.summarize_history()
    
    # 2. High Tariff (20%) on all tradable sectors
    params = create_symmetric_parameters()
    sim_shock = bootstrap_simulator(params)
    sim_shock.run(5) # burn in
    
    # Apply 20% tariff on H
    tariffs = {2: 0.2, 3: 0.2} # Assuming 2,3 are tradable
    sim_shock.apply_import_tariff("H", tariffs)
    sim_shock.run(15)
    
    shock_res = sim_shock.summarize_history()
    
    # Compare Income and Prices
    base_inc = base_res["H"]["income"][-1]
    shock_inc = shock_res["H"]["income"][-1]
    
    base_price = base_res["H"]["price_mean"][-1]
    shock_price = shock_res["H"]["price_mean"][-1]
    
    logger.info(f"Baseline Income (t=20): {base_inc:.4f}")
    logger.info(f"Shock Income (t=20): {shock_inc:.4f}")
    logger.info(f"Diff Income: {shock_inc - base_inc:.4f}")
    
    logger.info(f"Baseline Price (t=20): {base_price:.4f}")
    logger.info(f"Shock Price (t=20): {shock_price:.4f}")
    logger.info(f"Diff Price: {shock_price - base_price:.4f}")

    if abs(shock_inc - base_inc) < 1e-4 and abs(shock_price - base_price) < 1e-4:
        logger.error("Simulation INSENSITIVE to large tariff!")
    else:
        logger.info("Simulation SENSITIVE to large tariff.")
        
    # Check Objective Values
    from analysis.optimization.objective import evaluate_snapshot_objective
    
    # Needs a fresh snapshot
    params = create_symmetric_parameters()
    sim_snap = bootstrap_simulator(params)
    sim_snap.run(5)
    
    # 1. Zero Tariff
    score_std_0 = evaluate_snapshot_objective(sim_snap, actor="H", import_tariffs=None, objective_type="standard")
    score_rel_0 = evaluate_snapshot_objective(sim_snap, actor="H", import_tariffs=None, objective_type="relative")
    
    # 2. 20% Tariff
    tariffs = {2: 0.2, 3: 0.2}
    score_std_high = evaluate_snapshot_objective(sim_snap, actor="H", import_tariffs=tariffs, objective_type="standard")
    score_rel_high = evaluate_snapshot_objective(sim_snap, actor="H", import_tariffs=tariffs, objective_type="relative")
    
    logger.info(f"STD Objective: 0%={score_std_0:.4f}, 20%={score_std_high:.4f}")
    logger.info(f"REL Objective: 0%={score_rel_0:.4f}, 20%={score_rel_high:.4f}")
    
    if score_rel_high > score_rel_0:
        logger.info("Relative Objective FAVORS high tariff -> SPSA underperforming.")
    else:
        logger.info("Relative Objective DISLIKES high tariff -> Weights/Model issue.")
if __name__ == "__main__":
    test_sensitivity()
