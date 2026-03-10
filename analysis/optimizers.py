"""
优化器模块（Optimization Strategies）
======================================
本模块提供多种用于最优回应博弈的优化策略：
- BayesianOptimizer: 基于高斯过程的贝叶斯优化（全局搜索）
- SPSAOptimizer: 同时扰动随机逼近（适用于高维/噪声目标函数）
- GradientDescentOptimizer: 数值梯度上升法

所有优化器继承自 Optimizer 抽象基类，提供统一接口。

说明：
- 原文件名为 optimization.py，与 analysis/optimization/ 子包重名，会导致导入冲突。
- 现重命名为 optimizers.py；analysis/optimization/__init__.py 会继续向外 re-export 这些类，
  以兼容旧的 `from analysis.optimization import ...` 写法。
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np
import time

# bayesian-optimization 是可选依赖：缺失时仍允许使用 SPSA/GD
try:
    from bayes_opt import BayesianOptimization
    _HAS_BO = True
except ImportError:  # pragma: no cover
    BayesianOptimization = None  # type: ignore[assignment]
    _HAS_BO = False


class Optimizer(ABC):
    """
    优化器抽象基类。
    
    所有优化策略必须实现 optimize 方法，用于最大化目标函数。
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        n_iter: int = 20,
        **kwargs
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行优化，最大化目标函数。

        参数:
            objective_function: 目标函数，接受一维参数数组，返回标量分数。
            bounds: 参数边界，形状为 (n_params, 2)，每行为 [最小值, 最大值]。
            initial_guess: 可选的初始猜测点。
            n_iter: 迭代次数/步数。

        返回:
            best_params: 找到的最优参数。
            best_score: 最优目标值。
            history: 各迭代的目标值历史列表。
        """
        pass


class BayesianOptimizer(Optimizer):
    """
    贝叶斯优化器。
    
    使用高斯过程建模目标函数，通过采集函数平衡探索与利用。
    适用于评估成本高、低维度的黑盒优化问题。
    
    依赖: bayesian-optimization 库
    """
    
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        n_iter: int = 20,
        init_points: int = 5,
        **kwargs
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行贝叶斯优化。
        
        参数:
            objective_function: 目标函数。
            bounds: 参数边界。
            initial_guess: 初始探测点（可选）。
            n_iter: 总迭代次数（包含初始随机探测）。
            init_points: 初始随机探测点数量。
        
        返回:
            最优参数、最优值、历史记录。
        """
        if not _HAS_BO:
            raise RuntimeError(
                "未安装可选依赖 bayesian-optimization (bayes_opt)，无法使用 BayesianOptimizer。"
                "请先安装：pip install bayesian-optimization"
            )
        # 包装目标函数：bayes_opt 使用关键字参数，需要适配
        def bo_obj(**params):
            # bayes_opt 按字母序传递参数，我们用 p0, p1, ... 命名以保持顺序
            x = np.array([params[f'p{i}'] for i in range(len(params))])
            return objective_function(x)

        # 构建参数边界字典
        pbounds = {f'p{i}': (b[0], b[1]) for i, b in enumerate(bounds)}
        
        optimizer = BayesianOptimization(
            f=bo_obj,
            pbounds=pbounds,
            random_state=int(time.time()),
            verbose=0  # 静默模式
        )

        # 如果提供了初始猜测，先探测该点
        if initial_guess is not None:
            params_dict = {f'p{i}': val for i, val in enumerate(initial_guess)}
            optimizer.probe(
                params=params_dict,
                lazy=True,  # 延迟评估
            )

        # 执行优化
        optimizer.maximize(
            init_points=init_points,
            n_iter=max(1, n_iter - init_points),
        )

        # 提取最优结果
        best_params_dict = optimizer.max['params']
        best_params = np.array([best_params_dict[f'p{i}'] for i in range(len(best_params_dict))])
        best_score = optimizer.max['target']
        
        # 重构历史（按评估顺序的目标值）
        history = [res['target'] for res in optimizer.res]
        
        return best_params, best_score, history


class SPSAOptimizer(Optimizer):
    """
    同时扰动随机逼近优化器（SPSA）。
    
    SPSA 是一种无梯度优化方法，每次迭代仅需两次目标函数评估即可估计梯度。
    适用于高维、噪声大、梯度不可用的优化问题。
    
    参考文献: Spall, J.C. (1992). Multivariate stochastic approximation using 
              a simultaneous perturbation gradient approximation.
    """
    
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        n_iter: int = 20,
        learning_rate: float = 0.01,
        perturbation: float = 0.05,
        **kwargs
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行 SPSA 优化（梯度上升，最大化目标）。
        
        参数:
            objective_function: 目标函数。
            bounds: 参数边界。
            initial_guess: 初始点，默认为边界中心。
            n_iter: 迭代次数。
            learning_rate: 初始学习率 (a)。
            perturbation: 初始扰动幅度 (c)。
        
        返回:
            最优参数、最优值、历史记录。
        """
        dim = bounds.shape[0]
        
        # 初始化参数：使用初始猜测或边界中心
        if initial_guess is None:
            theta = np.mean(bounds, axis=1)
        else:
            theta = np.array(initial_guess)
            
        history = []
        best_theta = theta.copy()
        best_score = -float('inf')

        # SPSA 超参数衰减：a_k = a / (A + k)^alpha, c_k = c / k^gamma
        # 此处采用简化形式以提高稳定性
        
        for k in range(n_iter):
            # 评估当前点
            current_score = objective_function(theta)
            if current_score > best_score:
                best_score = current_score
                best_theta = theta.copy()
            history.append(current_score)
            
            # 生成伯努利扰动向量 (+1 或 -1)
            delta = 2 * np.round(np.random.rand(dim)) - 1
            
            # 计算衰减后的步长
            ck = perturbation / ((k + 1) ** 0.101)  # 扰动衰减
            ak = learning_rate / ((k + 1 + n_iter * 0.1) ** 0.602)  # 学习率衰减
            
            # 正负扰动点（裁剪到边界内）
            theta_plus = np.clip(theta + ck * delta, bounds[:, 0], bounds[:, 1])
            theta_minus = np.clip(theta - ck * delta, bounds[:, 0], bounds[:, 1])
            
            # 评估扰动点
            y_plus = objective_function(theta_plus)
            y_minus = objective_function(theta_minus)
            
            # 估计梯度（同时扰动形式）
            ghat = (y_plus - y_minus) / (2 * ck * delta)
            
            # 梯度上升更新（最大化目标）
            theta = theta + ak * ghat
            
            # 投影到可行域
            theta = np.clip(theta, bounds[:, 0], bounds[:, 1])

        # 最终检查
        final_score = objective_function(theta)
        if final_score > best_score:
            best_score = final_score
            best_theta = theta.copy()
        history.append(final_score)

        return best_theta, best_score, history


class GradientDescentOptimizer(Optimizer):
    """
    数值梯度上升优化器。
    
    使用有限差分法估计梯度，执行梯度上升以最大化目标函数。
    适用于低噪声、连续可微的目标函数。
    
    注意：每次迭代需要 2*dim 次函数评估（中心差分），计算成本较高。
    """
    
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        n_iter: int = 20,
        learning_rate: float = 0.01,
        epsilon: float = 1e-4,
        **kwargs
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行数值梯度上升优化。
        
        参数:
            objective_function: 目标函数。
            bounds: 参数边界。
            initial_guess: 初始点，默认为边界中心。
            n_iter: 迭代次数。
            learning_rate: 学习率。
            epsilon: 有限差分步长。
        
        返回:
            最优参数、最优值、历史记录。
        """
        dim = bounds.shape[0]
        
        # 初始化参数
        if initial_guess is None:
            theta = np.mean(bounds, axis=1)
        else:
            theta = np.array(initial_guess)
            
        history = []
        best_theta = theta.copy()
        best_score = -float('inf')
        
        for k in range(n_iter):
            # 评估当前点
            current_score = objective_function(theta)
            if current_score > best_score:
                best_score = current_score
                best_theta = theta.copy()
            history.append(current_score)
            
            # 计算数值梯度（中心差分）
            grad = np.zeros(dim)
            for i in range(dim):
                # 正向扰动
                theta_plus = theta.copy()
                theta_plus[i] += epsilon
                theta_plus[i] = min(theta_plus[i], bounds[i, 1])  # 裁剪到上界
                
                # 负向扰动
                theta_minus = theta.copy()
                theta_minus[i] -= epsilon
                theta_minus[i] = max(theta_minus[i], bounds[i, 0])  # 裁剪到下界
                
                # 评估
                y_plus = objective_function(theta_plus)
                y_minus = objective_function(theta_minus)
                
                # 中心差分梯度估计
                grad[i] = (y_plus - y_minus) / (theta_plus[i] - theta_minus[i] + 1e-10)
            
            # 梯度上升更新
            theta = theta + learning_rate * grad
            
            # 投影到可行域
            theta = np.clip(theta, bounds[:, 0], bounds[:, 1])
            
        return best_theta, best_score, history


__all__ = [
    "Optimizer",
    "BayesianOptimizer", 
    "SPSAOptimizer",
    "GradientDescentOptimizer",
    "HAS_BAYES_OPT",
]

# 对外暴露可选依赖状态，便于上层选择默认优化器
HAS_BAYES_OPT: bool = _HAS_BO
