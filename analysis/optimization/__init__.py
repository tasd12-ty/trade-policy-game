"""
策略/政策优化子包（analysis.optimization）
=========================================

该目录包含围绕两国动态经济仿真的政策优化与博弈实验代码，例如：
- objective.py：目标函数与基线/快照评估
- spsa_opt.py：SPSA/有限差分 PGD 黑箱优化器
- interaction.py：交互式（同时决策）博弈框架
- grad_game.py：可微分梯度博弈实验

注意：
- 顶层优化器策略已迁移到 `analysis/optimizers.py`。
- 为兼容旧写法，这里继续 re-export Optimizer 及其实现：
  `from analysis.optimization import BayesianOptimizer` 仍然可用。
"""

from analysis.optimizers import (
    Optimizer,
    BayesianOptimizer,
    SPSAOptimizer,
    GradientDescentOptimizer,
)

__all__ = [
    "Optimizer",
    "BayesianOptimizer",
    "SPSAOptimizer",
    "GradientDescentOptimizer",
]

