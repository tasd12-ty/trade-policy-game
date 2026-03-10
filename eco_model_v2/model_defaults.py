"""默认参数配置 — 集中管理所有未从数据校准的参数。

其他模块通过 import 引用此处的默认值。
当 io_params 管道未输出某参数时，使用此处的值。
用户可通过修改此文件或在调用时覆盖来调整参数。
"""

# ---- 生产端 ----

# Armington CES 弹性 ρ_ij（io_params 未输出，默认 Cobb-Douglas）
# ρ=0 对应 σ=1 (Cobb-Douglas)，ρ→−∞ 对应 Leontief
RHO_IJ_DEFAULT = 0.0

# 非贸易部门数 Ml（前 Ml 个部门不参与进口）
ML_DEFAULT = 0

# 要素投入弹性分配规则：
# "residual" — α_{i,Nl+k} = (1 − Σ_j α_{ij}) / M，平分给 M 个要素
FACTOR_ALPHA_RULE = "residual"

# ---- 消费端 ----

# 消费端 CES 弹性 ρ_cj（默认 Cobb-Douglas）
RHO_CJ_DEFAULT = 0.0

# ---- 动态参数 ----

# 价格调整速度 τ
TAU_DEFAULT = 0.3

# ---- 贸易参数 ----

# 默认进口成本乘数（无关税基准 = 1.0）
IMPORT_COST_DEFAULT = 1.0

# 默认出口量（无数据时假设 = 0.0）
EXPORT_DEFAULT = 0.0
