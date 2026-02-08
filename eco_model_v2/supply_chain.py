"""供应链扩展 (Section 3.3, eq 27)。

供应链通过函数邻接矩阵 F_{ij} 将不同国家的部门连接起来，
修改有效进口价格和出口数量。

设计：作为覆盖层(overlay)，不改变核心动态方程，
而是在每期开始时修改 imp_price 和 export 量。

公式对应：
- eq 27: 函数邻接矩阵 F_{G_s} = {F_{ij}: i,j ∈ K_s}
  其中 F_{ij}: R^2+ → R^2+ 将 (P_i, E_i) 映射到 (P_j, E_j)

依赖：utils.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .utils import EPS, clamp_positive


# ---- 供应链节点与边 ----

@dataclass
class SupplyChainNode:
    """供应链节点：对应某国某部门。

    属性：
        country: 国家标识 ("H" 或 "F")
        sector:  部门索引
    """
    country: str
    sector: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.country, self.sector)


@dataclass
class SupplyChainEdge:
    """供应链边：连接两个节点。

    F_{ij}(P_i, E_i) → (P_j^modified, E_j^modified)

    属性：
        source: 上游节点
        target: 下游节点
        transform: (P_source, E_source) → (P_target_adj, E_target_adj)
    """
    source: SupplyChainNode
    target: SupplyChainNode
    transform: Callable[[float, float], Tuple[float, float]]


@dataclass
class SupplyChainNetwork:
    """供应链网络 (eq 27)。

    由多条边组成，每条边定义上下游之间的价格和数量传导关系。
    在每期仿真开始时调用 apply() 修改进口价格和出口数量。
    """
    edges: List[SupplyChainEdge] = field(default_factory=list)

    def add_edge(
        self,
        source: SupplyChainNode,
        target: SupplyChainNode,
        transform: Callable[[float, float], Tuple[float, float]],
    ) -> None:
        """添加供应链边。"""
        self.edges.append(SupplyChainEdge(source, target, transform))

    def apply(
        self,
        home_price: np.ndarray,
        foreign_price: np.ndarray,
        home_export: np.ndarray,
        foreign_export: np.ndarray,
        home_imp_price: np.ndarray,
        foreign_imp_price: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """应用供应链效应。

        遍历所有边，根据上游节点的 (P, E) 计算下游节点的调整后 (P', E')。
        当多条边指向同一下游节点时，取加权平均。

        参数：
            home_price:      (Nl+M,) H 国产品+要素价格
            foreign_price:   (Nl+M,) F 国产品+要素价格
            home_export:     (Nl+M,) H 国出口量
            foreign_export:  (Nl+M,) F 国出口量
            home_imp_price:  (Nl,) H 国进口到岸价
            foreign_imp_price: (Nl,) F 国进口到岸价

        返回：
            dict 包含修改后的 imp_price 和 export
        """
        # 复制用于修改
        h_imp = np.array(home_imp_price, copy=True, dtype=float)
        f_imp = np.array(foreign_imp_price, copy=True, dtype=float)
        h_exp = np.array(home_export, copy=True, dtype=float)
        f_exp = np.array(foreign_export, copy=True, dtype=float)

        price_map = {"H": home_price, "F": foreign_price}
        export_map = {"H": home_export, "F": foreign_export}

        # 聚合多条边对同一目标的影响
        target_adjustments: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}

        for edge in self.edges:
            src = edge.source
            tgt = edge.target

            P_src = float(price_map[src.country][src.sector])
            E_src = float(export_map[src.country][src.sector])

            P_adj, E_adj = edge.transform(P_src, E_src)

            key = tgt.key
            if key not in target_adjustments:
                target_adjustments[key] = []
            target_adjustments[key].append((P_adj, E_adj))

        # 应用调整：加权平均
        for (country, sector), adjustments in target_adjustments.items():
            if not adjustments:
                continue

            # 加权平均价格（按数量加权）
            total_E = sum(e for _, e in adjustments)
            if total_E > EPS:
                avg_P = sum(p * e for p, e in adjustments) / total_E
            else:
                avg_P = np.mean([p for p, _ in adjustments])

            # 修改进口价格（下游面临的价格来自上游）
            if country == "H":
                h_imp[sector] = max(float(avg_P), EPS)
            else:
                f_imp[sector] = max(float(avg_P), EPS)

            # 修改上游出口量（eq 27 同时映射价格和数量）
            # 将变换后的总数量写回对应上游国家的出口
            for edge in self.edges:
                if edge.target.key == (country, sector):
                    src = edge.source
                    P_s = float(price_map[src.country][src.sector])
                    E_s = float(export_map[src.country][src.sector])
                    _, E_transformed = edge.transform(P_s, E_s)
                    if src.country == "H":
                        h_exp[src.sector] = max(float(E_transformed), 0.0)
                    else:
                        f_exp[src.sector] = max(float(E_transformed), 0.0)

        return {
            "home_imp_price": h_imp,
            "foreign_imp_price": f_imp,
            "home_export": h_exp,
            "foreign_export": f_exp,
        }


# ---- 常用供应链变换函数 ----

def identity_transform(P: float, E: float) -> Tuple[float, float]:
    """恒等变换（无效应）。"""
    return (P, E)


def markup_transform(markup: float = 0.1) -> Callable[[float, float], Tuple[float, float]]:
    """加成变换：P' = P × (1 + markup), E' = E。"""
    def transform(P: float, E: float) -> Tuple[float, float]:
        return (P * (1.0 + markup), E)
    return transform


def bottleneck_transform(
    capacity: float = 1.0,
    price_elasticity: float = 2.0,
) -> Callable[[float, float], Tuple[float, float]]:
    """瓶颈变换：当需求超过产能时价格上涨。

    P' = P × (1 + max(0, E/capacity - 1) × elasticity)
    E' = min(E, capacity)
    """
    def transform(P: float, E: float) -> Tuple[float, float]:
        excess = max(0.0, E / max(capacity, EPS) - 1.0)
        P_new = P * (1.0 + excess * price_elasticity)
        E_new = min(E, capacity)
        return (P_new, E_new)
    return transform


def disruption_transform(
    severity: float = 0.5,
) -> Callable[[float, float], Tuple[float, float]]:
    """供应链中断：价格上涨，数量缩减。

    P' = P / (1 - severity)
    E' = E × (1 - severity)
    """
    s = np.clip(severity, 0.0, 0.99)
    def transform(P: float, E: float) -> Tuple[float, float]:
        return (P / (1.0 - s), E * (1.0 - s))
    return transform
