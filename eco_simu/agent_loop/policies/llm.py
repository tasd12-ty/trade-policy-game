from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict

from ..actions import _validate_action_strict

PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class PolicyAdapter:
    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class LLMPolicyAdapter(PolicyAdapter):
    """核心LLM策略适配器 - 简化版
    
    流程:
    1. 构建提示词 (经济指标 -> 自然语言)
    2. 调用LLM API
    3. 解析响应并返回动作
    """

    def __init__(
        self,
        n_sectors: int = 5,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        timeout: float = 30.0,
        max_sectors_per_type: int = None,
        log_dir: str = None,
    ):
        self.n = n_sectors
        self.model = model
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.temperature = temperature
        self.maxn = max_sectors_per_type

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # 核心流程：观测 -> 提示词 -> LLM -> 动作
        actor = obs.get("country", "H")
        prompt = self._build_prompt(obs)
        payload = self._chat_payload(prompt)
        
        try:
            # 调用LLM
            raw_response = self._post(payload)
            response_obj = json.loads(raw_response)
            
            # 提取内容
            message = response_obj["choices"][0]["message"]
            tool_calls = message.get("tool_calls", [])
            
            # 解析JSON
            if tool_calls:
                json_str = tool_calls[0]["function"]["arguments"]
            else:
                json_str = message["content"]
            
            data = json.loads(json_str)
            action = _validate_action_strict(data, n_sectors=self.n, max_per_type=self.maxn)
            return action
        except Exception as e:
            raise RuntimeError(f"LLM policy failed for {actor}: {str(e)}")

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        # 核心提示词构造：经济状态 -> 策略决策指令
        metrics = obs["metrics"]
        delta = obs.get("metrics_delta", {})
        prev = obs.get("prev_action", {})
        actor = obs["country"]
        reward_weights = obs["reward_weights"]
        
        w_income = reward_weights["w_income"]
        w_price = reward_weights["w_price"]
        w_trade = reward_weights["w_trade"]

        prompt = f"""You are country {actor}. Make your next policy decision.
Return JSON: {{"type":"policy_bundle", "actor":"{actor}", "import_tariff":{{}}, "export_quota":{{}}, "import_multiplier":{{}}}}

Objective: Maximize reward r = {w_income}*income_growth - {w_price}*price_mean + {w_trade}*trade_balance

Current state:
- income: {metrics['income_last']}
- price_mean: {metrics['price_mean_last']}
- trade_balance: {metrics['trade_balance_last']}
- prev_action: {prev}

Constraints:
- Sector indices: 0..{self.n-1}
- import_tariff: discrete values [0.0, 0.1, 0.2, 0.3]
- export_quota: discrete values [1.0, 0.9, 0.8]
- Max sectors per policy type: {self.maxn}
"""
        return prompt


    def _chat_payload(self, prompt: str) -> Dict[str, Any]:
        # 核心API调用结构
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Reply with strict JSON object only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }

    def _post(self, payload: Dict[str, Any]) -> str:
        # 核心HTTP调用
        import urllib.request as _urlrequest
        import urllib.parse as _urlparse
        
        url = _urlparse.urljoin(self.base_url, "chat/completions")
        body = json.dumps(payload).encode("utf-8")
        req = _urlrequest.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        
        with _urlrequest.urlopen(req) as resp:
            return resp.read().decode("utf-8")



def build_policy_from_spec(spec: str, n_sectors: int) -> PolicyFn:
    """策略工厂 - 简化版"""
    if spec.startswith("llm"):
        model = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        
        return LLMPolicyAdapter(
            n_sectors=n_sectors,
            model=model,
            base_url=base,
            api_key=api_key,
            temperature=temp,
        )
    
    if spec.startswith("search"):
        # 核心搜索策略导入
        from .search import SearchPolicyAdapter
        return SearchPolicyAdapter(n_sectors=n_sectors)
    
    raise ValueError(f"Unknown policy spec: {spec}")


__all__ = ["PolicyAdapter", "LLMPolicyAdapter", "build_policy_from_spec"]
