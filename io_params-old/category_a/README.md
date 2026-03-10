# A 类参数计算说明（IO 直接识别）

## 范围

本目录用于从 OECD 投入产出表直接计算 A 类参数：

- `alpha_ij`
- `theta_ij`
- `beta_j`
- `theta_cj`

代码入口：`io_params/category_a/run.py`

---

## 公式

1. 生产投入系数 `alpha_ij`

`alpha_ij = Z_total[j,i] / OUTPUT[i]`

说明：IO 表原始中间投入矩阵是 `Z[j,i]`（行=供给部门 `j`，列=使用部门 `i`），输出结果转为模型方向（行 `i`，列 `j`）。

2. 生产端国内份额 `theta_ij`

`theta_ij = Z_dom[j,i] / Z_total[j,i]`

3. 消费权重 `beta_j`

`beta_j = C_total[j] / sum_j C_total[j]`

其中 `C_total[j]` 为选定最终需求列（默认 `HFCE, NPISH, GGFC`）在产品 `j` 上的合计。

4. 消费端国内份额 `theta_cj`

`theta_cj = C_dom[j] / C_total[j]`

---

## 数据来源

1. OECD 行业-行业投入产出 CSV（total + domestic）  
默认根目录：`中美投入产出表数据（附表格阅读说明）`

2. 表内字段

- 中间投入：`TTL_*` / `DOM_*` 行与 `D*` 列
- 总产出：`OUTPUT` 行
- 最终需求列：默认 `HFCE, NPISH, GGFC`（可在 CLI 覆盖）

---

## 输出文件

- `alpha_ij.csv`
- `theta_ij.csv`
- `beta_j.csv`
- `theta_cj.csv`
- `metadata.json`

默认输出目录：`io_params/outputs/category_a/<COUNTRYYEAR>/`

---

## 运行

```bash
python3 -m io_params.category_a.run \
  --country USA \
  --year 2022 \
  --data-root "中美投入产出表数据（附表格阅读说明）" \
  --out-dir io_params/outputs/category_a
```

