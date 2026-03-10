# C 类参数计算（外生情景路径）

## 范围

C 类参数用于动态情景输入，不由模型内部最优化求解。当前已实现：

- `A_i`（基线生产率）
- `gamma_ij`（生产端 Armington 权重）
- `gamma_cj`（消费端 Armington 权重）
- `rho_cj`（消费端 CES 形状参数）
- `Export_t`（外需路径）
- `p_t^O`（进口价格路径，输出文件名 `P_t_O.csv`）

## 计算公式

1. `A_i`

- `A_i = A_i^{ext}`（外生给定，默认 1，可按部门覆盖）

2. `gamma_ij`

- 先按既有口径映射：`gamma_ij := theta_ij`
- 再应用外生覆盖（若提供）
- 数值裁剪到 `[clip_min, clip_max]`

3. `gamma_cj`

- 先按既有口径映射：`gamma_cj := theta_cj`
- 再应用外生覆盖（若提供）
- 数值裁剪到 `[clip_min, clip_max]`

4. `rho_cj`

- `rho_cj = rho_cj^{ext}`（外生给定，默认 1，可按部门覆盖；内部做数值清洗）

5. `Export_t`

- `Export_t[j] = Export_base_j[j] * m_t^{export}[j]`
- 其中 `Export_base_j` 与 `m_t^{export}` 均来自统一外生输入文件（默认均为 1）

6. `p_t^O`

- `p_t^O[j] = p_{base,j}^O * m_t^{import}[j]`
- 其中 `p_{base,j}^O` 与 `m_t^{import}` 均来自统一外生输入文件（默认均为 1）

## 数据来源

- OECD IO 表：用于 `theta_ij`、`theta_cj` 的基础映射
- 外生配置：`external_inputs_default.json`（情景路径、基线、覆盖值）
- 可选基线 CSV：`--export-base-csv`、`--p-t-o-base-csv`

## 代码结构

- `external_inputs.py`: 外生输入读取（统一入口）
- `a_i.py`, `gamma_ij.py`, `gamma_cj.py`, `rho_cj.py`, `export_t.py`, `p_t_o.py`: 每个参数独立文件
- `pipeline.py`: C 类计算编排
- `run.py`: CLI 入口

## 运行方式

```bash
python3 -m io_params.category_c.run \
  --country USA \
  --year 2022 \
  --external-inputs io_params/category_c/external_inputs_default.json \
  --out-dir io_params/outputs/category_c
```

可选地接入已计算的基线向量：

```bash
python3 -m io_params.category_c.run \
  --country USA \
  --year 2022 \
  --export-base-csv io_params/outputs/category_b/USA2022/E_j.csv \
  --p-t-o-base-csv io_params/outputs/category_b/USA2022/P_j_O.csv
```
