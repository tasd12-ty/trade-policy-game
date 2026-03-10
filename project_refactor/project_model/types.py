"""核心数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CountryParams:
    """单国参数块。

    字段与论文/旧实现一一对应：
    - alpha[i,j]: 部门 i 使用部门 j 投入的产出弹性
    - gamma[i,j], rho[i,j]: 生产端 Armington 参数
    - beta[j]: 消费预算权重
    - A[i]: TFP
    - exports[j]: 基准出口量
    - gamma_cons[j], rho_cons[j]: 消费端 Armington 参数
    - import_cost[j]: 进口到岸价乘子
    """

    alpha: Array
    gamma: Array
    rho: Array
    beta: Array
    A: Array
    exports: Array
    gamma_cons: Array
    rho_cons: Array
    import_cost: Array

    def __post_init__(self) -> None:
        n = int(np.asarray(self.beta).shape[0])
        _check_shape(self.alpha, (n, n), "alpha")
        _check_shape(self.gamma, (n, n), "gamma")
        _check_shape(self.rho, (n, n), "rho")
        _check_shape(self.beta, (n,), "beta")
        _check_shape(self.A, (n,), "A")
        _check_shape(self.exports, (n,), "exports")
        _check_shape(self.gamma_cons, (n,), "gamma_cons")
        _check_shape(self.rho_cons, (n,), "rho_cons")
        _check_shape(self.import_cost, (n,), "import_cost")

    @property
    def n(self) -> int:
        return int(np.asarray(self.beta).shape[0])


@dataclass(frozen=True)
class TwoCountryParams:
    """两国模型参数。"""

    home: CountryParams
    foreign: CountryParams
    tradable_idx: Array

    def __post_init__(self) -> None:
        if self.home.n != self.foreign.n:
            raise ValueError("home 与 foreign 部门数必须一致")
        t = np.asarray(self.tradable_idx, dtype=int)
        if t.ndim != 1:
            raise ValueError("tradable_idx 必须是一维数组")
        if np.any(t < 0) or np.any(t >= self.home.n):
            raise ValueError("tradable_idx 含越界索引")

    @property
    def n(self) -> int:
        return self.home.n

    @property
    def tradable_mask(self) -> Array:
        mask = np.zeros(self.n, dtype=bool)
        mask[np.asarray(self.tradable_idx, dtype=int)] = True
        return mask

    @property
    def non_tradable_idx(self) -> Array:
        mask = self.tradable_mask
        return np.asarray([i for i in range(self.n) if not mask[i]], dtype=int)


@dataclass(frozen=True)
class CountryBlock:
    """静态均衡结果中的单国分块。"""

    X_dom: Array
    X_imp: Array
    C_dom: Array
    C_imp: Array
    price: Array
    imp_price: Array


@dataclass(frozen=True)
class StaticEquilibriumResult:
    """静态均衡求解结果。"""

    home: CountryBlock
    foreign: CountryBlock
    converged: bool
    iterations: int
    final_residual: float
    solver_message: str


@dataclass
class CountryState:
    """动态仿真状态。"""

    X_dom: Array
    X_imp: Array
    C_dom: Array
    C_imp: Array
    price: Array
    imp_price: Array
    export_base: Array
    export_actual: Array
    output: Array
    income: float

    def copy(self) -> "CountryState":
        """深拷贝状态，避免副作用。"""
        return CountryState(
            X_dom=np.array(self.X_dom, copy=True),
            X_imp=np.array(self.X_imp, copy=True),
            C_dom=np.array(self.C_dom, copy=True),
            C_imp=np.array(self.C_imp, copy=True),
            price=np.array(self.price, copy=True),
            imp_price=np.array(self.imp_price, copy=True),
            export_base=np.array(self.export_base, copy=True),
            export_actual=np.array(self.export_actual, copy=True),
            output=np.array(self.output, copy=True),
            income=float(self.income),
        )


def _check_shape(arr: Iterable[float], shape: tuple[int, ...], name: str) -> None:
    a = np.asarray(arr, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} 形状错误，期望 {shape}，实际 {a.shape}")
