# Eco-Model 代码概览

两国动态经济博弈仿真框架，支持 LLM 智能体策略与梯度优化最优响应求解。

---

## 目录结构

### `eco_simu/` — 经济仿真核心模块

```
eco_simu/
├── __init__.py              # 模块导出
├── model.py                 # 经济模型：生产函数、效用、均衡求解
├── sim.py                   # 动态仿真器（TwoCountrySimulator）
├── service.py               # 远程服务封装（API 调用）
├── plotting.py              # 结果可视化（历史轨迹、诊断图）
└── agent_loop/              # 智能体决策循环
    ├── __init__.py
    ├── actions.py           # 行动空间定义（关税、配额）
    ├── observations.py      # 观测构造（状态摘要）
    ├── reward.py            # 奖励函数（收入、贸易差额、稳定性）
    ├── workflow.py          # 决策循环主流程
    └── policies/            # 策略实现
        ├── __init__.py
        ├── base.py          # 策略基类
        ├── fixed.py         # 固定策略（无动作/预设）
        ├── llm.py           # LLM 智能体策略
        ├── random.py        # 随机策略
        └── search.py        # 搜索优化策略（SPSA/BO）
```

---

### `analysis/` — 研究分析与优化实验

```
analysis/
├── __init__.py
├── model/                   # 独立可微仿真实现（用于梯度优化）
│   ├── __init__.py
│   ├── model.py             # 经济模型（与 eco_simu 对齐，支持自动微分）
│   ├── sim.py               # 可微动态仿真器
│   └── smooth_ops.py        # 平滑化工具函数
│
├── optimization/            # 优化实验脚本
│   ├── __init__.py
│   ├── objective.py         # 目标函数定义
│   ├── grad_game.py         # 梯度博弈实验（两国同步决策）
│   ├── run_comparative_game.py  # 对比实验运行器
│   ├── run_demo.py          # 演示脚本
│   ├── interaction.py       # 交互分析
│   ├── param_sweep.py       # 参数扫描
│   ├── plotting.py          # 优化结果可视化
│   ├── spsa_opt.py          # SPSA 优化器封装
│   ├── tune_spsa.py         # SPSA 超参调优
│   └── debug_sensitivity.py # 敏感性调试
│
├── optimizers.py            # 优化器抽象（Bayesian/SPSA/GD）
├── game.py                  # 最优响应博弈框架
├── compare_optimizers.py    # 优化器对比实验
├── pipeline_demo.py         # 仿真流程演示
├── run_game.py              # 博弈运行入口
├── profile_speed.py         # 性能分析脚本
│
├── latex/                   # 数学推导文档（TeX）
└── results/                 # 实验结果输出
```

---

## 快速开始

```bash
# 运行梯度博弈实验
python analysis/optimization/grad_game.py

# 运行 SPSA 优化器博弈
python analysis/run_game.py --optimizer spsa --steps 200

# 仿真演示
python analysis/pipeline_demo.py
```

---

## 相关文档

- `production_network_simulation0916.pdf` — 经济模型理论推导
- `analysis/latex/bo_response_math.tex` — 最优响应目标函数定义

## 参数计算目录（按 A/B/C/D 分类）

- `io_params/`：新目录，按类别组织参数计算代码  
  - `io_params/category_a/`：A 类（IO 直接识别）
  - `io_params/category_b/`：B 类（IO + 外部输入，起步版）
  - `io_params/category_b/external_inputs_default.json`：外部参数集中配置，默认值均为 1
- `io_params_a/`：A 类旧路径兼容入口
