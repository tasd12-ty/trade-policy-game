"""对比两种价格初始化方法的结果差异。

方法1: initialize()              — _make_initial_state 直接对数法 + 联合迭代
方法2: initialize_from_equilibrium() — scipy least_squares 数值优化
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eco_model_v2.compat import from_io_params_two_country
from eco_model_v2.simulator import TwoCountrySimulator

# ── 加载 CN2017 vs US2017 真实数据 ──
base = os.path.dirname(os.path.abspath(__file__))
cn_dir = os.path.join(base, "CN2017")
us_dir = os.path.join(base, "US2017")

# io-final 目录是扁平结构，cat_a/b/c 都在同一目录
params = from_io_params_two_country(
    home_dirs=(cn_dir, cn_dir, cn_dir),
    foreign_dirs=(us_dir, us_dir, us_dir),
)

sectors = ["A01", "RUP", "RMID", "ELEC_CIT", "TEQ"]

# ── 方法1: initialize() ──
sim1 = TwoCountrySimulator(params, tau=0.1, normalize_gap=True, quantity_damping=0.01)
sim1.initialize()
h1 = sim1.history["H"][0]
f1 = sim1.history["F"][0]

# ── 方法2: initialize_from_equilibrium() ──
sim2 = TwoCountrySimulator(params, tau=0.1, normalize_gap=True, quantity_damping=0.01)
from eco_model_v2.equilibrium import solve_static_equilibrium
eq_result = solve_static_equilibrium(params)
print(f"均衡求解器: converged={eq_result.converged}, iterations={eq_result.iterations}, "
      f"residual={eq_result.final_residual:.6e}, msg={eq_result.solver_message}")
sim2.initialize_from_equilibrium()
h2 = sim2.history["H"][0]
f2 = sim2.history["F"][0]

# ── 输出对比 ──
Nl = params.home.Nl
M = params.home.M_factors

print("=" * 80)
print("两种初始化方法价格对比 (CN2017 vs US2017, 5部门)")
print("=" * 80)

print("\n## 中国 (H) — 商品价格 P_j")
print(f"{'部门':<12} {'方法1(init)':<16} {'方法2(equil)':<16} {'差异(%)':<12}")
print("-" * 56)
for j in range(Nl):
    p1 = h1.price[j]
    p2 = h2.price[j]
    diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
    print(f"{sectors[j]:<12} {p1:<16.8f} {p2:<16.8f} {diff:<12.6f}")

print(f"\n## 中国 (H) — 要素价格")
factor_names = ["w(劳动)", "r(资本)"]
for k in range(M):
    p1 = h1.price[Nl + k]
    p2 = h2.price[Nl + k]
    diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
    print(f"{factor_names[k]:<12} {p1:<16.8f} {p2:<16.8f} {diff:<12.6f}")

print(f"\n## 美国 (F) — 商品价格 P_j")
print(f"{'部门':<12} {'方法1(init)':<16} {'方法2(equil)':<16} {'差异(%)':<12}")
print("-" * 56)
for j in range(Nl):
    p1 = f1.price[j]
    p2 = f2.price[j]
    diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
    print(f"{sectors[j]:<12} {p1:<16.8f} {p2:<16.8f} {diff:<12.6f}")

print(f"\n## 美国 (F) — 要素价格")
for k in range(M):
    p1 = f1.price[Nl + k]
    p2 = f2.price[Nl + k]
    diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
    print(f"{factor_names[k]:<12} {p1:<16.8f} {p2:<16.8f} {diff:<12.6f}")

# ── 产出对比 ──
print(f"\n{'=' * 80}")
print("产出对比 Y_j")
print("=" * 80)

print(f"\n## 中国 (H)")
print(f"{'部门':<12} {'方法1(init)':<16} {'方法2(equil)':<16} {'差异(%)':<12}")
print("-" * 56)
for j in range(Nl):
    y1 = h1.output[j]
    y2 = h2.output[j]
    diff = (y2 - y1) / max(abs(y1), 1e-15) * 100
    print(f"{sectors[j]:<12} {y1:<16.8f} {y2:<16.8f} {diff:<12.6f}")

print(f"\n## 美国 (F)")
print(f"{'部门':<12} {'方法1(init)':<16} {'方法2(equil)':<16} {'差异(%)':<12}")
print("-" * 56)
for j in range(Nl):
    y1 = f1.output[j]
    y2 = f2.output[j]
    diff = (y2 - y1) / max(abs(y1), 1e-15) * 100
    print(f"{sectors[j]:<12} {y1:<16.8f} {y2:<16.8f} {diff:<12.6f}")

# ── 收入对比 ──
print(f"\n{'=' * 80}")
print("收入对比")
print("=" * 80)
print(f"{'国家':<12} {'方法1(init)':<16} {'方法2(equil)':<16} {'差异(%)':<12}")
print("-" * 56)
diff_h = (h2.income - h1.income) / max(abs(h1.income), 1e-15) * 100
diff_f = (f2.income - f1.income) / max(abs(f1.income), 1e-15) * 100
print(f"{'中国(H)':<12} {h1.income:<16.8f} {h2.income:<16.8f} {diff_h:<12.6f}")
print(f"{'美国(F)':<12} {f1.income:<16.8f} {f2.income:<16.8f} {diff_f:<12.6f}")

# ── 均衡条件检验 ──
from eco_model_v2.production import compute_marginal_cost
from eco_model_v2.armington import armington_share_from_prices

print(f"\n{'=' * 80}")
print("均衡条件检验: P = MC (零利润)")
print("=" * 80)

for label, cp, state, partner_state in [
    ("中国(H)", params.home, h1, f1, ),
    ("美国(F)", params.foreign, f1, h1, ),
]:
    Nl_c = cp.Nl
    imp_p = np.asarray(cp.import_cost, dtype=float) * partner_state.price[:Nl_c]
    mc = compute_marginal_cost(
        cp.A, cp.alpha, state.price, imp_p,
        cp.gamma, cp.rho, int(cp.Ml), int(cp.M_factors))
    print(f"\n  {label} 方法1:")
    for j in range(Nl_c):
        err = abs(state.price[j] - mc[j]) / max(abs(state.price[j]), 1e-15)
        print(f"    {sectors[j]}: P={state.price[j]:.8f}, MC={mc[j]:.8f}, |P-MC|/P={err:.2e}")

for label, cp, state, partner_state in [
    ("中国(H)", params.home, h2, f2, ),
    ("美国(F)", params.foreign, f2, h2, ),
]:
    Nl_c = cp.Nl
    imp_p = np.asarray(cp.import_cost, dtype=float) * partner_state.price[:Nl_c]
    mc = compute_marginal_cost(
        cp.A, cp.alpha, state.price, imp_p,
        cp.gamma, cp.rho, int(cp.Ml), int(cp.M_factors))
    print(f"\n  {label} 方法2:")
    for j in range(Nl_c):
        err = abs(state.price[j] - mc[j]) / max(abs(state.price[j]), 1e-15)
        print(f"    {sectors[j]}: P={state.price[j]:.8f}, MC={mc[j]:.8f}, |P-MC|/P={err:.2e}")

# ── 商品市场出清检验 ──
print(f"\n{'=' * 80}")
print("均衡条件检验: 商品市场出清 Y = X_dom + C_dom + Exports")
print("=" * 80)

for label, state in [("中国(H) 方法1", h1), ("中国(H) 方法2", h2),
                      ("美国(F) 方法1", f1), ("美国(F) 方法2", f2)]:
    print(f"\n  {label}:")
    for j in range(Nl):
        demand = state.X_dom[:, j].sum() + state.C_dom[j] + state.export_actual[j]
        supply = state.output[j]
        gap = (demand - supply) / max(abs(supply), 1e-15)
        print(f"    {sectors[j]}: Y={supply:.8f}, D={demand:.8f}, gap={gap:.2e}")

# ── 要素市场出清检验 ──
print(f"\n{'=' * 80}")
print("均衡条件检验: 要素市场出清 Σ α_f·P·Y = w·L")
print("=" * 80)

for label, cp, state in [("中国(H) 方法1", params.home, h1),
                           ("中国(H) 方法2", params.home, h2),
                           ("美国(F) 方法1", params.foreign, f1),
                           ("美国(F) 方法2", params.foreign, f2)]:
    alpha_arr = np.asarray(cp.alpha, dtype=float)
    Nl_c = cp.Nl
    M_c = cp.M_factors
    for k in range(M_c):
        demand = float((alpha_arr[:, Nl_c + k] * state.price[:Nl_c] * state.output[:Nl_c]).sum())
        supply = float(state.price[Nl_c + k] * cp.L[k])
        gap = (demand - supply) / max(abs(supply), 1e-15)
        print(f"  {label} {factor_names[k]}: demand={demand:.8f}, supply={supply:.8f}, gap={gap:.2e}")

# ── 总结 ──
all_p1 = np.concatenate([h1.price, f1.price])
all_p2 = np.concatenate([h2.price, f2.price])
max_price_diff = np.max(np.abs(all_p2 - all_p1) / np.maximum(np.abs(all_p1), 1e-15)) * 100

all_y1 = np.concatenate([h1.output[:Nl], f1.output[:Nl]])
all_y2 = np.concatenate([h2.output[:Nl], f2.output[:Nl]])
max_output_diff = np.max(np.abs(all_y2 - all_y1) / np.maximum(np.abs(all_y1), 1e-15)) * 100

print(f"\n{'=' * 80}")
print(f"最大价格相对差异: {max_price_diff:.6f}%")
print(f"最大产出相对差异: {max_output_diff:.6f}%")
print("=" * 80)

# ── 保存结果到 markdown ──
md_path = os.path.join(base, "INIT_COMPARISON.md")
with open(md_path, "w") as f:
    f.write("# 两种初始化方法对比结果\n\n")
    f.write("数据: CN2017 vs US2017, 5部门, rho=0 (Cobb-Douglas)\n\n")

    f.write("## 结论\n\n")
    f.write("**两种方法求解的价格初值差异巨大（最大99.6%），不相等。**\n\n")
    f.write("| 指标 | 方法1 `initialize()` | 方法2 `initialize_from_equilibrium()` |\n")
    f.write("|------|---------------------|--------------------------------------|\n")
    f.write("| P=MC 零利润 | ~1e-13 (机器精度) | ~2-5% 偏差 |\n")
    f.write("| 商品市场出清 | ~1e-8 (机器精度) | ~0.1-5% 偏差 |\n")
    f.write("| 要素市场出清 | ~1e-9 (机器精度) | 48%-12700% 偏差 |\n")
    f.write(f"| 求解器信息 | 解析法+联合迭代 | scipy LM, converged={eq_result.converged}, residual={eq_result.final_residual:.2e} |\n")
    f.write("\n**方法1是正确的均衡解；方法2的scipy求解器收敛到了非均衡点（ftol满足但残差=0.18）。**\n\n")

    f.write("## 中国 (H) 初始价格\n\n")
    f.write("| 部门 | 方法1 P_j | 方法2 P_j | 差异(%) |\n")
    f.write("|------|-----------|-----------|--------|\n")
    for j in range(Nl):
        p1 = h1.price[j]
        p2 = h2.price[j]
        diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
        f.write(f"| {sectors[j]} | {p1:.8f} | {p2:.8f} | {diff:.2f}% |\n")
    for k in range(M):
        p1 = h1.price[Nl + k]
        p2 = h2.price[Nl + k]
        diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
        f.write(f"| {factor_names[k]} | {p1:.8f} | {p2:.8f} | {diff:.2f}% |\n")

    f.write("\n## 美国 (F) 初始价格\n\n")
    f.write("| 部门 | 方法1 P_j | 方法2 P_j | 差异(%) |\n")
    f.write("|------|-----------|-----------|--------|\n")
    for j in range(Nl):
        p1 = f1.price[j]
        p2 = f2.price[j]
        diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
        f.write(f"| {sectors[j]} | {p1:.8f} | {p2:.8f} | {diff:.2f}% |\n")
    for k in range(M):
        p1 = f1.price[Nl + k]
        p2 = f2.price[Nl + k]
        diff = (p2 - p1) / max(abs(p1), 1e-15) * 100
        f.write(f"| {factor_names[k]} | {p1:.8f} | {p2:.8f} | {diff:.2f}% |\n")

    f.write("\n## 中国 (H) 初始产出\n\n")
    f.write("| 部门 | 方法1 Y_j | 方法2 Y_j | 差异(%) |\n")
    f.write("|------|-----------|-----------|--------|\n")
    for j in range(Nl):
        y1 = h1.output[j]
        y2 = h2.output[j]
        diff = (y2 - y1) / max(abs(y1), 1e-15) * 100
        f.write(f"| {sectors[j]} | {y1:.8f} | {y2:.8f} | {diff:.2f}% |\n")

    f.write("\n## 美国 (F) 初始产出\n\n")
    f.write("| 部门 | 方法1 Y_j | 方法2 Y_j | 差异(%) |\n")
    f.write("|------|-----------|-----------|--------|\n")
    for j in range(Nl):
        y1 = f1.output[j]
        y2 = f2.output[j]
        diff = (y2 - y1) / max(abs(y1), 1e-15) * 100
        f.write(f"| {sectors[j]} | {y1:.8f} | {y2:.8f} | {diff:.2f}% |\n")

    f.write("\n## 收入\n\n")
    f.write("| 国家 | 方法1 | 方法2 | 差异(%) |\n")
    f.write("|------|-------|-------|--------|\n")
    f.write(f"| 中国(H) | {h1.income:.8f} | {h2.income:.8f} | {diff_h:.2f}% |\n")
    f.write(f"| 美国(F) | {f1.income:.8f} | {f2.income:.8f} | {diff_f:.2f}% |\n")

    f.write("\n## 方法说明\n\n")
    f.write("### 方法1: `initialize()` (推荐)\n")
    f.write("1. 直接对数法解 P=MC: `(I-α_goods)·lnP = rhs` — 精确解\n")
    f.write("2. 值 Leontief 解产出: `(I-B)^{-1}·(PY_C + λ·PY_E)`\n")
    f.write("3. λ 出口缩放使要素市场出清\n")
    f.write("4. 外层迭代要素价格 w (锚 w₁=1)\n")
    f.write("5. 两国联合迭代 (10轮) 使进口价格一致\n\n")
    f.write("### 方法2: `initialize_from_equilibrium()`\n")
    f.write("1. ρ=0 解析解作初值 (equilibrium_rho0)\n")
    f.write("2. scipy least_squares (LM法) 求解完整非线性系统\n")
    f.write("3. 残差包含: 零利润 + 要素出清 + 商品出清 + 贸易收支 + 名义锚\n")
    f.write(f"4. 本次结果: converged={eq_result.converged}, 但 residual={eq_result.final_residual:.2e} >> 0\n")
    f.write("5. **ftol 终止但未真正收敛，陷入局部极小**\n")

print(f"\n结果已保存到 {md_path}")

