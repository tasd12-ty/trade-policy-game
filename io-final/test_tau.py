"""测试不同 tau 值对动态仿真稳定性和收敛速度的影响。

用 CN2017 vs US2017 真实数据，比较:
- 稳定性: 1000步后价格是否发散
- 收敛速度: 多快回到均衡附近
- 价格漂移: 最终价格相对初始的偏移
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eco_model_v2.compat import from_io_params_two_country
from eco_model_v2.simulator import TwoCountrySimulator

base = os.path.dirname(os.path.abspath(__file__))
cn_dir = os.path.join(base, "CN2017")
us_dir = os.path.join(base, "US2017")

params = from_io_params_two_country(
    home_dirs=(cn_dir, cn_dir, cn_dir),
    foreign_dirs=(us_dir, us_dir, us_dir),
)

sectors = ["A01", "RUP", "RMID", "ELEC", "TEQ"]
Nl = params.home.Nl
STEPS = 500

tau_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

print("=" * 100)
print(f"tau 参数敏感性测试 (CN2017 vs US2017, {STEPS}步)")
print("配置: normalize_gap=True, quantity_damping=0.01")
print("=" * 100)

results = []

for tau in tau_values:
    sim = TwoCountrySimulator(
        params, tau=tau, normalize_gap=True, quantity_damping=0.01)
    sim.initialize()

    p0_h = sim.history["H"][0].price[:Nl].copy()
    p0_f = sim.history["F"][0].price[:Nl].copy()
    y0_h = sim.history["H"][0].output[:Nl].copy()
    y0_f = sim.history["F"][0].output[:Nl].copy()

    stable = True
    diverge_step = None
    try:
        for step in range(STEPS):
            sim._step()
            ph = sim.history["H"][-1].price[:Nl]
            pf = sim.history["F"][-1].price[:Nl]
            # 检查是否发散 (价格超过初始的100倍或小于1/100)
            if np.any(ph / p0_h > 100) or np.any(ph / p0_h < 0.01) or \
               np.any(pf / p0_f > 100) or np.any(pf / p0_f < 0.01):
                stable = False
                diverge_step = step
                break
            if np.any(np.isnan(ph)) or np.any(np.isnan(pf)):
                stable = False
                diverge_step = step
                break
    except Exception as e:
        stable = False
        diverge_step = step if 'step' in dir() else 0

    if stable:
        pf_h = sim.history["H"][-1].price[:Nl]
        pf_f = sim.history["F"][-1].price[:Nl]
        yf_h = sim.history["H"][-1].output[:Nl]
        yf_f = sim.history["F"][-1].output[:Nl]

        # 价格漂移
        drift_h = np.max(np.abs(pf_h / p0_h - 1.0)) * 100
        drift_f = np.max(np.abs(pf_f / p0_f - 1.0)) * 100
        # 产出漂移
        ydrift_h = np.max(np.abs(yf_h / y0_h - 1.0)) * 100
        ydrift_f = np.max(np.abs(yf_f / y0_f - 1.0)) * 100

        # 最后100步的最大步间变化 (收敛度)
        max_step_delta = 0.0
        for t in range(-min(100, STEPS), 0):
            p_prev = sim.history["H"][t-1].price[:Nl]
            p_curr = sim.history["H"][t].price[:Nl]
            step_d = np.max(np.abs(p_curr / p_prev - 1.0))
            max_step_delta = max(max_step_delta, step_d)

        results.append({
            "tau": tau, "stable": True,
            "drift_h": drift_h, "drift_f": drift_f,
            "ydrift_h": ydrift_h, "ydrift_f": ydrift_f,
            "step_delta": max_step_delta,
        })
    else:
        results.append({
            "tau": tau, "stable": False,
            "diverge_step": diverge_step,
        })

# ── 汇总表 ──
print(f"\n{'tau':<8} {'稳定?':<8} {'H价格漂移%':<14} {'F价格漂移%':<14} "
      f"{'H产出漂移%':<14} {'F产出漂移%':<14} {'末100步Δ':<14} {'备注'}")
print("-" * 100)
for r in results:
    if r["stable"]:
        note = ""
        if r["step_delta"] < 1e-8:
            note = "已收敛"
        elif r["step_delta"] < 1e-4:
            note = "接近收敛"
        else:
            note = "仍在调整"
        print(f"{r['tau']:<8.2f} {'是':<8} {r['drift_h']:<14.6f} {r['drift_f']:<14.6f} "
              f"{r['ydrift_h']:<14.6f} {r['ydrift_f']:<14.6f} {r['step_delta']:<14.2e} {note}")
    else:
        print(f"{r['tau']:<8.2f} {'否':<8} {'—':<14} {'—':<14} "
              f"{'—':<14} {'—':<14} {'—':<14} 第{r['diverge_step']}步发散")

# ── Walrasian 模式对比 (仅测几个关键 tau) ──
print(f"\n{'=' * 100}")
print(f"Walrasian 模式对比 (walrasian=True, {STEPS}步)")
print("=" * 100)

tau_w_values = [0.05, 0.1, 0.3]
results_w = []
for tau in tau_w_values:
    sim = TwoCountrySimulator(
        params, tau=tau, normalize_gap=True, quantity_damping=0.01,
        walrasian=True)
    sim.initialize()

    p0_h = sim.history["H"][0].price[:Nl].copy()
    p0_f = sim.history["F"][0].price[:Nl].copy()

    stable = True
    diverge_step = None
    try:
        for step in range(STEPS):
            sim._step()
            ph = sim.history["H"][-1].price[:Nl]
            pf = sim.history["F"][-1].price[:Nl]
            if np.any(ph / p0_h > 100) or np.any(ph / p0_h < 0.01) or \
               np.any(pf / p0_f > 100) or np.any(pf / p0_f < 0.01):
                stable = False
                diverge_step = step
                break
            if np.any(np.isnan(ph)) or np.any(np.isnan(pf)):
                stable = False
                diverge_step = step
                break
    except Exception as e:
        stable = False
        diverge_step = step if 'step' in dir() else 0

    if stable:
        pf_h = sim.history["H"][-1].price[:Nl]
        pf_f = sim.history["F"][-1].price[:Nl]

        drift_h = np.max(np.abs(pf_h / p0_h - 1.0)) * 100
        drift_f = np.max(np.abs(pf_f / p0_f - 1.0)) * 100

        max_step_delta = 0.0
        for t in range(-min(100, STEPS), 0):
            p_prev = sim.history["H"][t-1].price[:Nl]
            p_curr = sim.history["H"][t].price[:Nl]
            step_d = np.max(np.abs(p_curr / p_prev - 1.0))
            max_step_delta = max(max_step_delta, step_d)

        results_w.append({
            "tau": tau, "stable": True,
            "drift_h": drift_h, "drift_f": drift_f,
            "step_delta": max_step_delta,
        })
    else:
        results_w.append({
            "tau": tau, "stable": False,
            "diverge_step": diverge_step,
        })

print(f"\n{'tau':<8} {'稳定?':<8} {'H价格漂移%':<14} {'F价格漂移%':<14} "
      f"{'末100步Δ':<14} {'备注'}")
print("-" * 80)
for r in results_w:
    if r["stable"]:
        note = ""
        if r["step_delta"] < 1e-8:
            note = "已收敛"
        elif r["step_delta"] < 1e-4:
            note = "接近收敛"
        else:
            note = "仍在调整"
        print(f"{r['tau']:<8.2f} {'是':<8} {r['drift_h']:<14.6f} {r['drift_f']:<14.6f} "
              f"{r['step_delta']:<14.2e} {note}")
    else:
        print(f"{r['tau']:<8.2f} {'否':<8} {'—':<14} {'—':<14} "
              f"{'—':<14} 第{r['diverge_step']}步发散")
