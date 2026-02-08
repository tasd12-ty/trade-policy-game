"""LLM 智能体接口。

参考 grad_op/analysis/optimization/llm/ 目录设计。
定义策略智能体抽象接口与具体实现。

依赖：sandbox.py (仅类型引用)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import re


# ---- 数据结构 ----

@dataclass
class PolicyDecision:
    """策略决策结果。"""
    tariff: Dict[int, float]                  # {部门: 关税率}
    quota: Dict[int, float]                   # {部门: 配额乘子}
    reasoning: str = ""                       # 决策理由
    raw_response: str = ""                    # LLM 原始响应
    system_prompt: str = ""                   # 系统 prompt（审计用）
    user_prompt: str = ""                     # 用户 prompt（审计用）
    token_usage: Optional[Dict[str, int]] = None


@dataclass
class EconomicContext:
    """经济观测上下文（用于 prompt 构建）。"""
    country: str
    opponent: str
    round_num: int
    # 本国状态
    income: float = 0.0
    trade_balance: float = 0.0
    price_mean: float = 0.0
    prices: List[float] = field(default_factory=list)
    outputs: List[float] = field(default_factory=list)
    exports: List[float] = field(default_factory=list)
    # 对手状态
    opponent_income: float = 0.0
    opponent_price_mean: float = 0.0
    # 当前政策
    current_tariff: Dict[int, float] = field(default_factory=dict)
    current_quota: Dict[int, float] = field(default_factory=dict)
    # 对手上轮政策
    opponent_prev_tariff: Dict[int, float] = field(default_factory=dict)
    opponent_prev_quota: Dict[int, float] = field(default_factory=dict)
    # 约束
    active_sectors: List[int] = field(default_factory=list)
    max_tariff: float = 1.0
    min_quota: float = 0.0


# ---- 抽象基类 ----

class PolicyAgent(ABC):
    """策略智能体抽象基类。"""

    @abstractmethod
    def observe(self, context: Dict) -> None:
        """接收经济观测。"""
        ...

    @abstractmethod
    def decide(self) -> Dict[str, Any]:
        """返回决策 {"tariff": {...}, "quota": {...}}。"""
        ...


# ---- 固定策略 ----

class FixedPolicyAgent(PolicyAgent):
    """固定策略智能体：始终返回相同的决策。"""

    def __init__(
        self,
        tariff: Dict[int, float] | None = None,
        quota: Dict[int, float] | None = None,
    ):
        self.tariff = tariff or {}
        self.quota = quota or {}

    def observe(self, context: Dict) -> None:
        pass  # 不使用观测

    def decide(self) -> Dict[str, Any]:
        return {"tariff": dict(self.tariff), "quota": dict(self.quota)}


# ---- 以牙还牙策略 ----

class TitForTatAgent(PolicyAgent):
    """以牙还牙策略：复制对手上一轮的政策。

    观测中的 import_cost 反映了对手施加的关税（cost > base 表示有关税）。
    决策时以对手的 import_cost 变化推断其关税率，然后模仿。
    """

    def __init__(
        self,
        initial_tariff: Dict[int, float] | None = None,
        base_import_cost: float = 1.1,
    ):
        self._tariff = dict(initial_tariff or {})
        self._quota: Dict[int, float] = {}
        self._base_cost = base_import_cost
        self._last_obs: Dict | None = None

    def observe(self, context: Dict) -> None:
        self._last_obs = context
        # 从对手的 price vs 本国 import_cost 推断对手关税
        # opponent_price 变化间接反映了对手的关税效应
        # 简化实现：从 import_cost 推断 tariff = (cost / base) - 1
        import_costs = context.get("import_cost", [])
        inferred_tariff = {}
        for j, cost in enumerate(import_costs):
            rate = max(0.0, float(cost) / max(self._base_cost, 1e-9) - 1.0)
            if rate > 0.01:
                inferred_tariff[j] = rate
        self._tariff = inferred_tariff

    def decide(self) -> Dict[str, Any]:
        return {"tariff": dict(self._tariff), "quota": dict(self._quota)}


# ---- LLM 策略 ----

class LLMPolicyAgent(PolicyAgent):
    """LLM 智能体策略。

    使用 LLM API 进行策略决策。

    参数：
        llm_client:   LLM 客户端（需实现 generate 方法）
        active_sectors: 可调整的部门列表
        max_tariff:   关税上限
        min_quota:    配额下限
        temperature:  LLM 温度
        max_tokens:   最大 token 数
        max_retries:  最大重试次数
    """

    def __init__(
        self,
        llm_client: Any,
        active_sectors: List[int] | None = None,
        max_tariff: float = 1.0,
        min_quota: float = 0.0,
        temperature: float | None = None,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ):
        self.llm_client = llm_client
        self.active_sectors = active_sectors or []
        self.max_tariff = max_tariff
        self.min_quota = min_quota
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self._context: Dict | None = None
        self._round_num: int = 0
        self._history: List[Dict] = []

    def observe(self, context: Dict) -> None:
        """接收经济观测。"""
        self._context = context
        self._round_num += 1

    def decide(self) -> Dict[str, Any]:
        """使用 LLM 生成策略决策。"""
        if self._context is None:
            return {"tariff": {}, "quota": {}}

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()

        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                raw_text = response if isinstance(response, str) else response.get("text", "")
                parsed = self._extract_json(raw_text)

                if parsed is not None:
                    tariff, quota = self._validate_policy(parsed)
                    decision = PolicyDecision(
                        tariff=tariff,
                        quota=quota,
                        reasoning=parsed.get("reasoning", ""),
                        raw_response=raw_text,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    self._history.append({
                        "round": self._round_num,
                        "decision": {"tariff": tariff, "quota": quota},
                        "reasoning": decision.reasoning,
                    })
                    return {"tariff": tariff, "quota": quota}

            except Exception:
                continue

        # 所有重试失败：保守策略
        return {"tariff": {}, "quota": {}}

    def _build_system_prompt(self) -> str:
        """构建系统 prompt。"""
        sectors_str = ", ".join(str(s) for s in self.active_sectors) if self.active_sectors else "所有可贸易部门"
        return f"""你是一个经济贸易政策专家 AI，代表一个国家在两国贸易博弈中制定政策。

## 你的目标
1. **收入增长**：最大化本国国民收入
2. **贸易差额**：改善贸易收支
3. **价格稳定**：维持物价稳定

## 政策工具
- **关税率** tariff: 每个部门的进口关税率，范围 [0, {self.max_tariff}]
- **出口配额** quota: 每个部门的出口配额乘子，范围 [{self.min_quota}, 1.0]
  (1.0 = 不限制, 0.5 = 出口减半, 0.0 = 完全禁止)

## 可调整部门
{sectors_str}

## 博弈规则
- 每轮你和对手同时做出决策
- 你的决策会影响进口价格和出口量
- 对手也会根据经济状况调整政策

## 输出格式
请用 JSON 格式回答：
```json
{{
  "reasoning": "你的决策理由（简要）",
  "tariff": {{"部门号": 关税率, ...}},
  "quota": {{"部门号": 配额乘子, ...}}
}}
```"""

    def _build_user_prompt(self) -> str:
        """构建用户 prompt。"""
        ctx = self._context
        if ctx is None:
            return "无可用观测数据。"

        lines = [
            f"## 当前状态（第 {self._round_num} 回合）",
            f"",
            f"### 本国 ({ctx.get('country', '?')})",
            f"- 国民收入: {ctx.get('income', 0):.4f}",
            f"- 贸易差额: {ctx.get('trade_balance', 0):.4f}",
            f"- 产品价格: {ctx.get('price', [])}",
            f"- 部门产出: {ctx.get('output', [])}",
            f"- 出口量: {ctx.get('exports', [])}",
            f"",
            f"### 对手国",
            f"- 国民收入: {ctx.get('opponent_income', 0):.4f}",
            f"- 产品价格: {ctx.get('opponent_price', [])}",
            f"",
            f"### 当前政策",
            f"- 进口成本乘子: {ctx.get('import_cost', [])}",
            f"",
            f"请根据以上信息制定本轮政策。",
        ]

        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从 LLM 响应中提取 JSON。"""
        # 尝试 1: 直接解析
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试 2: 提取 markdown 代码块
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

        # 尝试 3: 提取大括号内容
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _validate_policy(
        self,
        parsed: Dict,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """验证并钳位政策。"""
        tariff = {}
        raw_tariff = parsed.get("tariff", {})
        for k, v in raw_tariff.items():
            sector = int(k)
            rate = float(v)
            tariff[sector] = max(0.0, min(rate, self.max_tariff))

        quota = {}
        raw_quota = parsed.get("quota", {})
        for k, v in raw_quota.items():
            sector = int(k)
            mult = float(v)
            quota[sector] = max(self.min_quota, min(mult, 1.0))

        return tariff, quota
