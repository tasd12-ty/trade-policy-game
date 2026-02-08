"""核心数据结构定义。

维度约定（与论文一致）：
- Nl: 生产部门数（产品种类数）
- Ml: 非贸易部门数（部门 0..Ml-1 不可贸易，Ml..Nl-1 可贸易）
- M:  要素种类数（如劳动、资本）

价格与产出向量长度为 Nl+M：
- price[0..Nl-1] 为产品价格
- price[Nl..Nl+M-1] 为要素价格
- output[0..Nl-1] 为部门产出
- output[Nl..Nl+M-1] 为要素禀赋（固定常数）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Array = np.ndarray


# ---- 参数结构 ----

@dataclass(frozen=True)
class CountryParams:
    """单国参数块（含要素市场）。

    字段说明：
    - alpha: (Nl, Nl+M)  部门 i 使用投入 j 的产出弹性
        列 0..Ml-1:    非贸易中间品投入
        列 Ml..Nl-1:   可贸易中间品投入
        列 Nl..Nl+M-1: 要素投入（劳动、资本等）
    - gamma: (Nl, Nl)    生产端 Armington 国内权重 γ_ij
    - rho:   (Nl, Nl)    生产端 Armington 形状参数 ρ_ij
    - beta:  (Nl,)       消费预算权重
    - A:     (Nl,)       全要素生产率 TFP
    - exports: (Nl+M,)   基准出口量（后 M 项为 0，要素不出口）
    - gamma_cons: (Nl,)  消费端 Armington 国内权重
    - rho_cons:   (Nl,)  消费端 Armington 形状参数
    - import_cost: (Nl,) 进口到岸价乘子（含关税/运输）
    - L:     (M,)        要素禀赋（劳动人数、资本存量等，固定常数）
    - Ml:    int          非贸易部门数
    - M_factors: int      要素种类数
    """
    alpha: Array          # (Nl, Nl+M)
    gamma: Array          # (Nl, Nl)
    rho: Array            # (Nl, Nl)
    beta: Array           # (Nl,)
    A: Array              # (Nl,)
    exports: Array        # (Nl+M,)
    gamma_cons: Array     # (Nl,)
    rho_cons: Array       # (Nl,)
    import_cost: Array    # (Nl,)
    L: Array              # (M,)
    Ml: int               # 非贸易部门数
    M_factors: int        # 要素种类数

    def __post_init__(self) -> None:
        """维度校验。"""
        Nl = int(np.asarray(self.beta).shape[0])
        M = int(self.M_factors)
        Ml = int(self.Ml)
        _check((Nl, Nl + M), self.alpha, "alpha")
        _check((Nl, Nl), self.gamma, "gamma")
        _check((Nl, Nl), self.rho, "rho")
        _check((Nl,), self.beta, "beta")
        _check((Nl,), self.A, "A")
        _check((Nl + M,), self.exports, "exports")
        _check((Nl,), self.gamma_cons, "gamma_cons")
        _check((Nl,), self.rho_cons, "rho_cons")
        _check((Nl,), self.import_cost, "import_cost")
        _check((M,), self.L, "L")
        if Ml < 0 or Ml > Nl:
            raise ValueError(f"Ml 必须在 [0, Nl={Nl}] 范围内，实际 {Ml}")

    @property
    def Nl(self) -> int:
        """生产部门数。"""
        return int(np.asarray(self.beta).shape[0])

    @property
    def tradable_mask(self) -> Array:
        """可贸易部门布尔掩码 (Nl,)。"""
        mask = np.zeros(self.Nl, dtype=bool)
        mask[self.Ml:] = True
        return mask

    @property
    def tradable_idx(self) -> Array:
        """可贸易部门索引。"""
        return np.arange(self.Ml, self.Nl, dtype=int)

    @property
    def nontradable_idx(self) -> Array:
        """非贸易部门索引。"""
        return np.arange(0, self.Ml, dtype=int)

    @property
    def n_tradable(self) -> int:
        """可贸易部门数 = Nl - Ml。"""
        return self.Nl - self.Ml

    @property
    def value_added_share(self) -> Array:
        """增加值份额 v_i = 1 - Σ_j α_{ij} (Nl,)。"""
        return np.clip(1.0 - np.asarray(self.alpha, dtype=float).sum(axis=1),
                       1e-8, None)


@dataclass(frozen=True)
class TwoCountryParams:
    """两国模型参数。"""
    home: CountryParams
    foreign: CountryParams

    def __post_init__(self) -> None:
        if self.home.Nl != self.foreign.Nl:
            raise ValueError("两国部门数 Nl 必须一致")
        if self.home.Ml != self.foreign.Ml:
            raise ValueError("两国非贸易部门数 Ml 必须一致")
        if self.home.M_factors != self.foreign.M_factors:
            raise ValueError("两国要素数 M 必须一致")

    @property
    def Nl(self) -> int:
        return self.home.Nl

    @property
    def Ml(self) -> int:
        return self.home.Ml

    @property
    def M(self) -> int:
        return self.home.M_factors


# ---- 结果结构 ----

@dataclass(frozen=True)
class CountryBlock:
    """静态均衡结果中的单国分块。"""
    X_dom: Array       # (Nl, Nl+M) 国内中间品 + 要素使用量
    X_imp: Array       # (Nl, Nl) 进口中间品（仅可贸易列非零）
    C_dom: Array       # (Nl,) 国内消费
    C_imp: Array       # (Nl,) 进口消费（仅可贸易部门非零）
    price: Array       # (Nl+M,) 产品价格 + 要素价格
    imp_price: Array   # (Nl,) 进口品到岸价
    output: Array      # (Nl,) 各部门产出
    income: float      # 消费者收入


@dataclass(frozen=True)
class StaticEquilibriumResult:
    """静态均衡求解结果。"""
    home: CountryBlock
    foreign: CountryBlock
    converged: bool
    iterations: int
    final_residual: float
    solver_message: str


# ---- 动态状态 ----

@dataclass
class CountryState:
    """动态仿真中的单国状态快照。"""
    X_dom: Array         # (Nl, Nl+M) 国内中间品 + 要素使用量
    X_imp: Array         # (Nl, Nl) 进口中间品
    C_dom: Array         # (Nl,) 国内消费
    C_imp: Array         # (Nl,) 进口消费
    price: Array         # (Nl+M,) 产品+要素价格
    imp_price: Array     # (Nl,) 进口品到岸价
    export_base: Array   # (Nl+M,) 基准出口量
    export_actual: Array # (Nl+M,) 当期实际出口量
    output: Array        # (Nl+M,) 产出 + 要素禀赋
    income: float        # 消费者收入

    def copy(self) -> CountryState:
        """深拷贝，避免副作用。"""
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


# ---- 内部工具 ----

def _check(shape: tuple[int, ...], arr: Iterable[float], name: str) -> None:
    """维度校验辅助。"""
    a = np.asarray(arr, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} 形状错误：期望 {shape}，实际 {a.shape}")
