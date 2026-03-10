# B 类参数计算说明（IO + 外部输入，联立求解增强版）

## 范围

本目录用于计算 B 类参数：

- `w, r, K, L`
- `P_j, P_j^O`
- `E_j`

代码入口：`io_params/category_b/run.py`

参数函数拆分（与 `category_a` 风格一致）：

- `io_params/category_b/factor_params.py`
  - `compute_w`, `compute_r`, `compute_k`, `compute_l`, `compute_factor_params`
- `io_params/category_b/p_j.py`
  - `compute_p_j_exogenous`, `compute_p_j_from_endogenous_solution`
- `io_params/category_b/p_j_o.py`
  - `compute_p_j_o_exogenous`, `compute_p_j_o_from_endogenous_solution`
- `io_params/category_b/export_value_j.py`
  - `compute_export_value_j`
- `io_params/category_b/e_j.py`
  - `compute_e_j`
- `io_params/category_b/price_system.py`
  - `build_equilibrium_raw_params`, `solve_price_system_endogenous`
- `io_params/category_b/pipeline.py`
  - 轻量编排 `compute_b_parameters`

函数约束：

- 各参数函数均为纯函数设计，不修改输入对象，不依赖全局可变状态。
- 运行顺序可交换（除联立求解需要的显式依赖输入外），重复调用结果一致。

---

## 当前实现公式

1. 要素价格/存量

- `w = LaborIncome / L`
- `r = external_interest_rate`
- `K = CapitalIncome / r`
- `L = external_labor_population`

2. 国内/进口价格（`P_j, P_j^O`）

- `endogenous` 模式（默认）：
  - 用 A 类参数（`alpha_ij, theta_ij, beta_j, theta_cj`）构造静态均衡输入
  - 调用 `eco_simu.model.solve_initial_equilibrium` 联立求解价格系统
  - 读取求解结果中的 `P_j` 与 `P_j^O`
- `exogenous` 模式：
  - `P_j`：优先取分部门外部给定值；缺失时用 `domestic_price_default`
  - `P_j^O`：优先取分部门外部给定值；缺失时用 `P_j * import_price_multiplier_default`
- 若 `endogenous` 模式失败且 `fallback_to_exogenous_if_solver_fails=1`，自动回退到 `exogenous`。

3. 基期出口量

- `E_j = ExportValue_j / P_j`
- `ExportValue_j` 优先读取 IO 表最终需求列 `EXPO`，缺失时使用 `export_value_fallback`

---

## 数据来源

1. OECD IO 表（total/domestic）  
用于读取部门集合与 `EXPO`（若存在）。

2. 外部给定参数（统一文件）

- `io_params/category_b/external_inputs_default.json`

该文件是 B 类全部外部输入的集中配置文件，默认数值均为 `1`。

---

## 外部参数字段

- `labor_income_total`
- `labor_population`
- `interest_rate_r`
- `capital_income_total`
- `price_mode`（`endogenous` 或 `exogenous`）
- `domestic_price_default`
- `import_price_multiplier_default`
- `export_value_fallback`
- `rho_default`
- `tfp_A_default`
- `import_cost_default`
- `use_all_sectors_tradable`
- `fallback_to_exogenous_if_solver_fails`
- `domestic_prices_by_sector`（可选分部门覆盖）
- `import_prices_by_sector`（可选分部门覆盖）

---

## 输出文件

- `factor_params.csv`（`w,r,K,L`）
- `P_j.csv`
- `P_j_O.csv`
- `E_j.csv`
- `ExportValue_j.csv`
- `diagnostics.json`
- `metadata.json`

默认输出目录：`io_params/outputs/category_b/<COUNTRYYEAR>/`

---

## 运行

```bash
python3 -m io_params.category_b.run \
  --country USA \
  --year 2022 \
  --data-root "中美投入产出表数据（附表格阅读说明）" \
  --external-inputs io_params/category_b/external_inputs_default.json \
  --price-mode auto \
  --solver-max-iterations 400 \
  --solver-tolerance 1e-8 \
  --out-dir io_params/outputs/category_b
```
