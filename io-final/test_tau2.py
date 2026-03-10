"""测试无 quantity_damping 时 tau 的稳定边界。

quantity_damping=1.0 (无阻尼) 时，tau 的选择更关键。
"""
import sys, os
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

Nl = params.home.Nl
sectors = ["A01", "RUP", "RMID", "ELEC", "TEQ"]
STEPS = 500

configs = [
    ("normalize_gap=True, qd=1.0 (无数量阻尼)", True, 1.0, False),
    ("normalize_gap=False, qd=0.01", False, 0.01, False),
    ("normalize_gap=True, qd=0.01, walrasian", True, 0.01, True),
]

tau_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

for config_name, ng, qd, wal in configs:
    print(f"\n{'=' * 90}")
    print(f"配置: {config_name}, {STEPS}步")
    print("=" * 90)

    print(f"{'tau':<8} {'稳定?':<8} {'H价格漂移%':<14} {'F价格漂移%':<14} {'末100步Δ':<14} {'备注'}")
    print("-" * 72)

    for tau in tau_values:
        sim = TwoCountrySimulator(
            params, tau=tau, normalize_gap=ng, quantity_damping=qd,
            walrasian=wal)
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
                   np.any(pf / p0_f > 100) or np.any(pf / p0_f < 0.01) or \
                   np.any(np.isnan(ph)) or np.any(np.isnan(pf)):
                    stable = False
                    diverge_step = step
                    break
        except Exception:
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

            note = "已收敛" if max_step_delta < 1e-8 else \
                   "接近收敛" if max_step_delta < 1e-4 else "仍在调整"
            print(f"{tau:<8.2f} {'是':<8} {drift_h:<14.6f} {drift_f:<14.6f} "
                  f"{max_step_delta:<14.2e} {note}")
        else:
            print(f"{tau:<8.2f} {'否':<8} {'—':<14} {'—':<14} "
                  f"{'—':<14} 第{diverge_step}步发散")
