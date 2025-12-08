# eco-model 代码概览


## 代码文件树
```text
.
├── agent_loop.py                      # 多国智能体博弈循环与策略封装
├── api_server.py                      # 简易 API 服务入口
├── run_simulation.py                  # 纯仿真运行入口
├── run_experiment.py                  # 实验入口，含导出与绘图控制
├── eco_simu/                          # 经济仿真核心模块
│   ├── agent_loop/                    # 观测、奖励、行动、策略定义
│   │   ├── actions.py
│   │   ├── observations.py
│   │   ├── reward.py
│   │   ├── workflow.py
│   │   └── policies/                  # LLM 与搜索策略
│   │       ├── llm.py
│   │       └── search.py
│   ├── model.py                       # 经济模型结构与参数
│   ├── sim.py                         # 仿真驱动
│   ├── service.py                     # 远程服务封装
│   └── plotting.py                    # 结果绘图
├── analysis/                          # 研究脚本与数学推导
│   ├── optimal_policy/                # SPSA/BO 等最优政策求解
│   │   ├── objective.py
│   │   ├── spsa_opt.py
│   │   ├── run_demo.py
│   │   └── bo_response_math.tex
│   └── standalone_sim/                # 独立仿真实现
│       ├── model.py
│       ├── sim.py
│       └── __init__.py
├── scripts/                           # 批量实验、搜索与 ETL 脚本
│   ├── run_configured_experiments.py
│   ├── search_grid_runner.py
│   ├── test_search_candidates.py
│   ├── etl_oecd_to_params.py
│   └── *.sh
├── eco-modelonly/                     # 历史仿真结果（图表/CSV）
├── production_network_simulation0916.tex | production_network_simulation0916.pdf
├── requirements.txt                   # 依赖列表
└── 中美投入产出表数据（附表格阅读说明）/   # 输入输出表数据（中美，OECD/官方）
```

