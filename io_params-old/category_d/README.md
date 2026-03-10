# D 类参数说明（拟合/优化校准）

## 范围

D 类参数不能直接“读表”，需要通过历史轨迹拟合或优化估计。典型包括：

- `epsilon`（进口需求价格弹性）
- `rho_ij`（生产端替代弹性，若不固定）
- `tau`（价格调整速度）
- `Omega_I*`（传统政策目标权重）
- `F: Context_t -> Omega_t^{II*}`（Agent 决策映射）

---

## 典型估计口径（目标形式）

1. 轨迹拟合

`min_theta sum_t || y_t^model(theta) - y_t^data ||^2 + R(theta)`

其中 `theta` 可包含 `tau, epsilon, rho_ij`。

2. 政策偏好权重拟合

`min_omega sum_t L(policy_t^observed, policy_t^model(omega | state_t))`

3. Agent 映射学习

`min_F sum_t L(Omega_t^pred(F(Context_t)), Omega_t^target)`

---

## 数据来源

- 历史贸易政策样本（2017-2025 等）
- 宏观与行业时序（贸易差额、GDP、产出、价格）
- 新闻/政治/情绪等上下文特征（用于 Agent 映射）

---

## 当前状态

本目录当前仅提供说明文档，尚未实现估计器。后续建议实现：

- `fit_tau.py`
- `fit_elasticity.py`
- `fit_policy_weights.py`
- `fit_context_mapping.py`

并统一输出：

- `estimates.json`
- `fit_report.md`
- `diagnostics/`（残差与稳健性图表）
