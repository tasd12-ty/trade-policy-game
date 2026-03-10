# 参数计算模块（按类别分层）

目录结构：

- `io_params/common/`: 公共常量与 OECD IO 表加载逻辑
- `io_params/category_a/`: A 类参数（可由 IO 表直接识别）
- `io_params/category_b/`: B 类参数（IO + 外部输入，支持联立求解）
- `io_params/category_c/`: C 类参数（外生情景路径 + 映射参数）
- `io_params/category_d/`: D 类占位目录

## A 类（已实现）

- `alpha_ij = Z_total[j,i] / OUTPUT[i]`
- `theta_ij = Z_dom[j,i] / Z_total[j,i]`
- `beta_j`: 最终消费支出占比归一化
- `theta_cj = C_dom[j] / C_total[j]`

运行：

```bash
python3 -m io_params.category_a.run \
  --country USA \
  --year 2022 \
  --data-root "中美投入产出表数据（附表格阅读说明）" \
  --out-dir io_params/outputs/category_a
```

## B 类（联立求解增强版）

当前实现：

- `w, r, K, L`（由外部输入计算）
- `P_j, P_j^O`（默认联立求解；失败可回退外生给定）
- `E_j = ExportValue_j / P_j`（ExportValue 优先取 IO 中 `EXPO` 列）

外部输入统一放在：

- `io_params/category_b/external_inputs_default.json`

该文件中可外生给定数值默认都为 `1`。

运行：

```bash
python3 -m io_params.category_b.run \
  --country USA \
  --year 2022 \
  --external-inputs io_params/category_b/external_inputs_default.json \
  --price-mode auto \
  --out-dir io_params/outputs/category_b
```

## C 类（已实现）

当前实现：

- `A_i`
- `gamma_ij`（按 `theta_ij` 映射 + 外生覆盖）
- `gamma_cj`（按 `theta_cj` 映射 + 外生覆盖）
- `rho_cj`
- `Export_t = Export_base_j * multiplier_t`
- `P_t_O = P_t_O_base_j * multiplier_t`

外部输入统一放在：

- `io_params/category_c/external_inputs_default.json`

该文件中外生值默认均为 `1`。

运行：

```bash
python3 -m io_params.category_c.run \
  --country USA \
  --year 2022 \
  --external-inputs io_params/category_c/external_inputs_default.json \
  --out-dir io_params/outputs/category_c
```
