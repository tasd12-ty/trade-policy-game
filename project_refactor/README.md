# Project_1 重构版框架（参考 `grad_op`）

## 目标

本目录提供一个**全新、解耦**的实现框架，按 `Project_1__LLM_Agents_as_Dynamic_Game_Players.pdf` 的核心公式重构：

- 静态均衡：生产/消费/贸易平衡联立求解
- 动态仿真：价格-需求-供给离散更新
- 政策接口：关税、进口乘子、出口配额
- 工程约束：模块解耦、中文注释、可测试

## 架构设计

```text
project_refactor/
  project_model/
    types.py         # 数据结构（参数、状态、结果）
    armington.py     # Armington/CES 数学函数
    production.py    # 产出、边际成本、收入
    equilibrium.py   # 静态均衡方程组与求解器
    dynamics.py      # 动态更新与两国仿真器
    policy.py        # 策略事件与时间线执行
    presets.py       # 参数模板与旧格式转换
    pipeline.py      # 高层启动入口
  examples/
    run_demo.py      # 最小运行示例
  tests/
    test_armington.py
    test_equilibrium.py
    test_dynamics.py
```

## 公式映射

1. 生产函数（生产端）

- `Y_i = A_i * Π_j component_{ij}`
- 可贸易部门投入使用 Armington 合成：
  - `component_{ij} = [gamma_ij * (X_ij^I)^rho_ij + (1-gamma_ij) * (X_ij^O)^rho_ij]^(alpha_ij/rho_ij)`

2. 对偶成本 / 零利润

- `log(lambda_i) = -log(A_i) + Σ_j alpha_ij * log(P_j^*)`
- 均衡约束：`P_i = lambda_i`

3. 消费需求

- 非贸易品：`C_j = beta_j * I / P_j`
- 贸易品：按 Armington 份额拆分 `C_j^I` 与 `C_j^O`

4. 动态价格更新

- `P_{t+1} = P_t * exp(tau * (D_t - Y_t))`
- 可选归一化：`(D_t - Y_t)/Y_t`

5. 供给与外汇约束

- 供不应求：按比例缩放国内用途与出口
- 进口受外汇约束：`s_fx = min(1, V_exp / V_imp)`

## 设计上的“解耦保障”

- 数学函数与经济逻辑分层：`armington.py` 不依赖求解器
- 静态求解与动态仿真分离：`equilibrium.py` / `dynamics.py`
- 策略时间线独立：`policy.py` 只调用仿真器公开接口
- 入口统一：`pipeline.py` 支持结构化参数和旧字典参数

## 精度与稳定性保障

- 全流程使用 `float64`
- 对数计算使用 `EPS` 下界保护
- 价格更新使用 `tanh` 截断避免指数爆炸
- 静态均衡采用 `log-parameterization + least_squares`

## 快速运行

先创建虚拟环境（已按本项目验证）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3 --system-site-packages project_refactor/.venv
source project_refactor/.venv/bin/activate
```

说明：当前沙箱网络受限时，使用 `--system-site-packages` 复用系统已有 `numpy`，可直接跑示例和测试。

然后运行示例：

```bash
python3 project_refactor/examples/run_demo.py
```

## 运行测试

```bash
python3 -m unittest discover -s project_refactor/tests -p "test_*.py"
```
