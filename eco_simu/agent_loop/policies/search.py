from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import sys
import time
import warnings
from itertools import product

from ..actions import apply_actions_to_sim, zeros_like_action
from ..observations import build_obs
from ..reward import compute_reward

from .llm import PolicyAdapter

"""
SearchPolicyAdapter：在克隆仿真器上演练候选动作，按配置化目标函数选择最优回应。
"""


class SearchPolicyAdapter(PolicyAdapter):
    """基于规则或数值优化的策略适配器。
    
    支持多种搜索方法 (method):
    - 'grid'：网格搜索（默认，也是唯一实现）。遍历用户指定的关税/配额“挡位”，在目标部门上做笛卡尔积（关税×配额×部门）。

    逻辑说明 (Game Logic):
    - 采用“独立最优回应” (Independent Best Response) 机制。
    - 假设对手在未来 k 步内**不会采取新行动** (Fixed Action / Inertia)。
    - target_sectors 必须明确提供；不再做“自动截断 + 候选上限”的简化。
    """

    def __init__(
        self,
        n_sectors: int,
        lookahead_rounds: int = 1,
        lookahead_steps: Optional[int] = None,
        tariff_grid: Optional[Sequence[float]] = None,
        quota_grid: Optional[Sequence[float]] = None,
        target_sectors: Optional[Sequence[int]] = None,
        objective: str = "reward",
        method: str = "grid",
    ):
        self.n = int(max(n_sectors, 1))
        self.lookahead_rounds = max(int(lookahead_rounds), 0)
        self.lookahead_steps = None if lookahead_steps is None else max(int(lookahead_steps), 0)
        self.tariff_grid = list(tariff_grid) if tariff_grid is not None else [0.05, -0.05, 0.1, -0.1]
        self.quota_grid = list(quota_grid) if quota_grid is not None else [0.8, 0.6]
        self.target_sectors = list(target_sectors) if target_sectors is not None else None
        self.objective = (objective or "reward").strip().lower()
        self.method = (method or "grid").strip().lower()

        self._sim: Any = None
        self._actor: Optional[str] = None
        self._reward_weights: Optional[Dict[str, float]] = None
        self._k_per_step: int = 1
        self._obs: Optional[Dict[str, Any]] = None  # 当前观测，用于优化过程中的评估

    def bind_runtime(
        self,
        sim: Any,
        actor: str,
        reward_weights: Optional[Dict[str, float]] = None,
        k_per_step: int = 1,
    ) -> None:
        """绑定运行期仿真器，便于执行“假设行动 → 推演”搜索。"""
        self._sim = sim
        self._actor = str(actor or "H").upper()
        self._reward_weights = dict(reward_weights) if reward_weights else None
        self._k_per_step = max(int(k_per_step), 1)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._sim is None or self._actor is None:
            raise RuntimeError("SearchPolicyAdapter must be bound via bind_runtime() before use")

        self._obs = obs  # 保存当前观测供评估使用
        actor = str(obs.get("country") or self._actor).upper()
        targets = self._select_targets(obs, actor)

        if self.method != "grid":
            warnings.warn(f"Method '{self.method}' not supported; defaulting to grid search.", UserWarning)
        return self._optimize_grid(actor, targets)

    # ----------------------------
    # 优化策略实现
    # ----------------------------

    def _optimize_grid(self, actor: str, targets: List[int]) -> Dict[str, Any]:
        """原有网格搜索逻辑"""
        candidates = self._build_candidates(actor, targets)
        diagnostics: List[Tuple[str, float]] = []
        best_score = -math.inf
        best_action: Optional[Dict[str, Any]] = None

        n_cand = len(candidates)
        print(f"[Search] Grid search started: {n_cand} candidates (targets={targets})")
        
        t0 = time.time()
        for i, action in enumerate(candidates):
            if i > 0 and i % 10 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-3)
                print(f"[Search] .. evaluated {i}/{n_cand} ({rate:.1f} cand/s)", end="\r", file=sys.stderr)

            try:
                score = self._evaluate_action(action)
            except Exception:
                continue
            diagnostics.append((self._summarize_action(action), float(score)))
            if score > best_score:
                best_score = float(score)
                best_action = action

        if best_action is None:
            best_action = self._zero_action(actor)
            best_score = float("-inf")

        action_out = copy.deepcopy(best_action)
        action_out["rationale"] = f"grid search objective={self.objective}; score={best_score:.6g}"
        if diagnostics:
            top_diag = ", ".join(f"{desc}:{val:.3g}" for desc, val in diagnostics[:5])
            action_out["plan"] = f"tested {len(diagnostics)} candidates -> {top_diag}"
        return action_out


    # ----------------------------
    # 核心评估逻辑
    # ----------------------------

    def _evaluate_action(self, action: Dict[str, Any]) -> float:
        """评估单个动作的预期收益"""
        fork = getattr(self._sim, "fork", None)
        cloner = getattr(self._sim, "clone", None)
        if callable(fork):
            sim_copy = fork(keep_history=False)
        elif callable(cloner):
            sim_copy = cloner()
        else:
            sim_copy = copy.deepcopy(self._sim)

        apply_actions_to_sim(sim_copy, actor=self._actor, action=action)
        
        steps = self._steps_to_simulate()
        for i in range(max(steps, 1)):
            sim_copy.step()
            
        fut_obs = build_obs(
            sim_copy,
            self._actor,
            prev_action=action,
            last_actions={self._actor: action},
            obs_full=False,
        )
        
        # 使用保存的 obs 中的权重，或者 adapter 自身的权重
        weights = self._reward_weights or (self._obs and self._obs.get("reward_weights")) or {"w_income": 1.0, "w_price": 1.0, "w_trade": 1.0}
        fut_obs["reward_weights"] = weights
        return self._objective_value(fut_obs, weights)

    # ----------------------------
    # 辅助方法
    # ----------------------------

    def _zero_action(self, actor: str) -> Dict[str, Any]:
        action = zeros_like_action()
        action["actor"] = actor
        return action

    def _select_targets(self, obs: Dict[str, Any], actor: str) -> List[int]:
        del obs, actor
        if self.target_sectors is None:
            raise ValueError("target_sectors must be provided for search; set sectors=... in policy spec")
        return [i for i in self.target_sectors if 0 <= i < self.n]

    def _build_candidates(self, actor: str, targets: Iterable[int]) -> List[Dict[str, Any]]:
        """生成跨部门、跨杠杆的笛卡尔积候选（关税×配额×部门），不再截断候选数量。"""
        targets_list = list(targets)
        if not targets_list:
            return [self._zero_action(actor)]

        per_sector_options: List[List[Dict[str, Dict[int, float]]]] = []
        for idx in targets_list:
            tariffs = [None]
            quotas = [None]
            for val in self.tariff_grid:
                try:
                    value = float(val)
                except Exception:
                    continue
                if not -1.0 <= value <= 1.0:
                    continue
                if abs(value) < 1e-9:
                    continue
                tariffs.append(value)
            for q in self.quota_grid:
                try:
                    qval = float(q)
                except Exception:
                    continue
                if not 0.0 <= qval <= 1.0:
                    continue
                if abs(qval - 1.0) < 1e-9:
                    continue
                quotas.append(qval)

            sector_actions: List[Dict[str, Dict[int, float]]] = []
            for t_val, q_val in product(tariffs, quotas):
                piece: Dict[str, Dict[int, float]] = {}
                if t_val is not None:
                    piece["import_tariff"] = {idx: float(t_val)}
                if q_val is not None:
                    piece["export_quota"] = {idx: float(q_val)}
                sector_actions.append(piece)
            per_sector_options.append(sector_actions)

        candidates: List[Dict[str, Any]] = []
        for combo in product(*per_sector_options):
            action = self._zero_action(actor)
            tariff_map: Dict[int, float] = {}
            quota_map: Dict[int, float] = {}
            for piece in combo:
                for k, v in (piece.get("import_tariff") or {}).items():
                    tariff_map[k] = v
                for k, v in (piece.get("export_quota") or {}).items():
                    quota_map[k] = v
            if tariff_map:
                action["import_tariff"] = tariff_map
            if quota_map:
                action["export_quota"] = quota_map
            candidates.append(action)

        uniq: List[Dict[str, Any]] = []
        seen: set = set()
        for cand in candidates:
            key = (
                tuple(sorted((cand.get("import_tariff") or {}).items())),
                tuple(sorted((cand.get("export_quota") or {}).items())),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(cand)
        return uniq

    def _steps_to_simulate(self) -> int:
        if self.lookahead_steps is not None:
            return max(int(self.lookahead_steps), 0)
        return max(self.lookahead_rounds * self._k_per_step, 1)

    def _objective_value(self, obs: Dict[str, Any], weights: Dict[str, float]) -> float:
        metrics = obs.get("metrics", {})
        if self.objective == "reward":
            return float(compute_reward(obs, weights=weights))
        if self.objective == "income":
            return float(metrics.get("income_last", 0.0))
        if self.objective == "trade":
            return float(metrics.get("trade_balance_last", 0.0))
        if self.objective == "price":
            return -float(metrics.get("price_mean_last", 0.0))
        return float(compute_reward(obs, weights=weights))

    def _summarize_action(self, action: Dict[str, Any]) -> str:
        tariff = action.get("import_tariff") or {}
        quota = action.get("export_quota") or {}
        if not tariff and not quota:
            return "noop"
        parts: List[str] = []
        if tariff:
            parts.append("tariff" + str(tariff))
        if quota:
            parts.append("quota" + str(quota))
        return "|".join(parts)


__all__ = ["SearchPolicyAdapter"]
