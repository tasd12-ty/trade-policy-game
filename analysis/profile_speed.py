"""
Profiling Script for Analysis Module
====================================
This script measures the execution speed of:
1. Single simulation step (TwoCountryDynamicSimulator.step)
2. Single agent decision step (OptimalResponseAgent.get_action) using SPSAOptimizer

Usage:
    python analysis/profile_speed.py
"""

import sys
import time
import numpy as np

from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model.model import create_symmetric_parameters, normalize_model_params, solve_initial_equilibrium
from analysis.model.sim import TwoCountryDynamicSimulator
from analysis.game import OptimalResponseAgent
from analysis.optimizers import SPSAOptimizer

def run_profiling():
    print("=== Profiling Setup ===")
    
    # 1. Setup Simulator
    print("Initializing simulator...")
    raw_params = create_symmetric_parameters()
    eq_result = solve_initial_equilibrium(raw_params)
    
    if not eq_result["convergence_info"]["converged"]:
        print("Error: Initial equilibrium failed to converge.")
        return

    sim = TwoCountryDynamicSimulator(
        normalize_model_params(raw_params), 
        eq_result, 
        theta_price=0.05
    )
    
    # Warmup
    print("Warming up simulator (100 steps)...")
    for _ in range(100):
        sim.step()
        
    print("\n=== 1. Simulation Step Speed ===")
    n_sim_steps = 1000
    times_sim = []
    
    print(f"Running {n_sim_steps} simulation steps...")
    for _ in range(n_sim_steps):
        t0 = time.time()
        sim.step()
        times_sim.append(time.time() - t0)
        
    avg_sim = np.mean(times_sim)
    std_sim = np.std(times_sim)
    print(f"Simulation Step Speed: {avg_sim*1000:.4f} ms/step (±{std_sim*1000:.4f} ms)")
    
    print("\n=== 2. Agent Decision Speed (SPSA) ===")
    
    n_sectors = sim.params.home.alpha.shape[0]
    agent = OptimalResponseAgent(
        "H", n_sectors, 
        optimizer=SPSAOptimizer(),
        lookahead_steps=25  # Standard lookahead
    )
    
    n_decision_steps = 10
    times_decision = []
    
    print(f"Running {n_decision_steps} agent decision steps (SPSA)...")
    # Clone sim state to avoid mutating standard flow too much, though get_action shouldn't mutate sim
    # actually get_action copies sim internally for planning
    
    for i in range(n_decision_steps):
        t0 = time.time()
        # get_action relies on sim state
        _ = agent.decide(sim, {}) 
        dt = time.time() - t0
        times_decision.append(dt)
        print(f"  Decision {i+1}: {dt:.4f} s")
        
    avg_dec = np.mean(times_decision)
    std_dec = np.std(times_decision)
    print(f"Agent Decision Speed: {avg_dec:.4f} s/step (±{std_dec:.4f} s)")
    
    print("\n=== Summary ===")
    print(f"Simulation (per step): {avg_sim*1000:.2f} ms")
    print(f"Decision (per step):   {avg_dec:.4f} s")
    print("Done.")

if __name__ == "__main__":
    run_profiling()
