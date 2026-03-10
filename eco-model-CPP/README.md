# eco-model-CPP

两国多部门动态经济仿真模型的 C++ 实现。

## 依赖

- **必需**: CMake 3.20+, Eigen 3.4+
- **可选**: Ceres Solver 2.2+ (均衡求解), nlohmann_json (JSON I/O)

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行示例

```bash
./build/examples/pipeline_demo
```

## 项目结构

```
include/eco_model/
├── core/           # 基本类型和参数结构
├── armington/      # Armington CES 函数
├── production/     # 生产函数、成本、收入
├── equilibrium/    # 静态均衡求解
├── simulation/     # 动态仿真
└── policy/         # 政策事件
```

## 对应 Python 代码

本项目重构自 `analysis/` 目录的 Python 实现，主要对应关系：

| Python | C++ |
|--------|-----|
| `model.py::CountryParams` | `core/country_params.hpp` |
| `model.py::armington_*` | `armington/ces.hpp` |
| `model.py::compute_output` | `production/output.hpp` |
| `sim.py::CountrySimulator` | `simulation/country_sim.hpp` |
| `sim.py::TwoCountryDynamicSimulator` | `simulation/two_country_sim.hpp` |
