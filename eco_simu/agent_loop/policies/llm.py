from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from urllib import error as _urlerror
from urllib import request as _urlrequest
from urllib.parse import urljoin

from ..actions import _validate_action_strict

PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]

"""
策略适配器：封装 LLM 请求、提示构造与解析，确保输出满足政策协议。
"""


class PolicyAdapter:
    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - 接口定义
        """策略接口：根据观测给出下一期动作。"""
        raise NotImplementedError


class LLMPolicyAdapter(PolicyAdapter):
    """基于本地 OpenAI 兼容接口的 LLM 策略适配器。
    
    核心机制:
    1. Prompt 工程: 将经济指标 (Income, Price, Trade) 转化为自然语言描述。
    2. 约束注入: 在 Prompt 中明确关税 [-1, 1]、配额 [0, 1] 等数值范围。
    3. 鲁棒性 (Ultrathink): 
       - 捕获 JSON 解析错误或数值越界错误。
       - 将错误信息反馈给 LLM 进行重试 (默认 5 次)。
       - 支持 Function Calling (Tools) 以获得更稳定的 JSON 输出。
    """

    def __init__(
        self,
        n_sectors: int = 5,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        timeout: float = 30.0,
        max_sectors_per_type: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        self.n = int(n_sectors)
        self.model = model
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key or "EMPTY"
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.maxn = int(max_sectors_per_type) if max_sectors_per_type is not None else None
        self._log_base = Path(log_dir or os.getenv("AGENT_LOG_DIR", "output/agent_logs")).absolute()
        self._run_tag = time.strftime("%Y%m%d-%H%M%S")
        self.ultra = str(os.getenv("AGENT_ULTRATHINK", "true")).strip().lower() in ("", "true", "1", "yes", "y")
        self.use_json_block = str(os.getenv("AGENT_JSON_BLOCK_MODE", "true")).strip().lower() in ("", "true", "1", "yes", "y")
        # 为提升结构化输出稳定性，优先使用 OpenAI 工具调用（function calling），可通过 AGENT_USE_TOOLS 关闭
        self.use_tools = str(
            os.getenv(
                "AGENT_USE_TOOLS",
                ("true" if self.use_json_block else "false"),
            )
        ).strip().lower() in ("", "true", "1", "yes", "y")

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        actor = str(obs.get("country", "H"))
        prompt = self._build_prompt(obs)
        base_payload = self._chat_payload(prompt)
        try:
            retries = int(os.getenv("AGENT_LLM_RETRIES", "5"))
        except Exception:
            retries = 2
        attempts = max(1, int(retries) + 1)

        last_error: Optional[Exception] = None
        raw_obj: Optional[Dict[str, Any]] = None
        content_preview: Optional[str] = None
        last_feedback: Optional[str] = None
        json_payload_for_log: Optional[str] = None

        for attempt in range(1, attempts + 1):
            req_payload = dict(base_payload)
            messages: List[Dict[str, Any]] = list(base_payload.get("messages", []))  # type: ignore[assignment]
            if attempt > 1:
                if self.ultra and last_feedback:
                    messages.append({"role": "user", "content": last_feedback})
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your last reply was invalid. Reply again with ONLY a strict JSON object for the same instruction. No commentary.",
                        }
                    )
            req_payload["messages"] = messages

            try:
                raw = self._post(req_payload)
                raw_obj = json.loads(raw)
                choices = raw_obj.get("choices") or []
                msg = choices[0]["message"] if choices else {}
                content = msg.get("content")
                tool_calls = msg.get("tool_calls") or []
                # 优先解析工具调用的 arguments（更稳定的 JSON 来源）
                if self.use_tools and tool_calls:
                    args_str = tool_calls[0].get("function", {}).get("arguments", "")
                    content_preview = (args_str or "")[:200]
                    json_payload_for_log = args_str
                    data = self._parse_json_strictish(args_str)
                else:
                    if not isinstance(content, str):
                        raise ValueError("LLM returned empty content")
                    content_preview = content[:200]
                    if self.use_json_block:
                        json_payload = self._extract_json_block(content)
                    else:
                        json_payload = content.strip()
                    json_payload_for_log = json_payload
                    data = self._parse_json_strictish(json_payload)
                action = self._to_action_strict(data, actor)
                self._write_log(
                    actor,
                    obs.get("t", -1),
                    {
                        "attempt": attempt,
                        "request": req_payload,
                        "response": raw_obj,
                        "parsed": action,
                        "json_payload_head": json_payload_for_log[:200] if json_payload_for_log else None,
                        "tool_used": bool(self.use_tools and tool_calls),
                        "ok": True,
                    },
                )
                return action
            except Exception as err:
                last_error = err
                try:
                    last_feedback = self._build_ultrathink_feedback(
                        err,
                        actor,
                        content_preview or "",
                        json_payload_for_log or "",
                        self.n,
                        self.maxn,
                    )
                except Exception:
                    last_feedback = None
                self._write_log(
                    actor,
                    obs.get("t", -1),
                    {
                        "attempt": attempt,
                        "request": req_payload,
                        "response": raw_obj,
                        "raw_content_head": content_preview,
                        "json_payload_head": json_payload_for_log[:200] if json_payload_for_log else None,
                        "error": str(err),
                        "ultrathink_feedback": last_feedback,
                        "ok": False,
                    },
                )
                if attempt < attempts:
                    continue
                raise last_error

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        metrics = obs.get("metrics", {})
        delta = obs.get("metrics_delta", {}) or {}
        prev = obs.get("prev_action", {}) or {}
        last_actions = obs.get("last_actions", {}) or {}
        vectors_summary = obs.get("vectors_summary", {}) or {}
        actor = str(obs.get("country", "H"))
        reward_weights = obs.get("reward_weights") or {"w_income": 1.0, "w_price": 1.0, "w_trade": 1.0}
        try:
            w_income = float(reward_weights.get("w_income", 1.0))
        except Exception:
            w_income = 1.0
        try:
            w_price = float(reward_weights.get("w_price", 0.0))
        except Exception:
            w_price = 0.0
        try:
            w_trade = float(reward_weights.get("w_trade", 0.0))
        except Exception:
            w_trade = 0.0

        lines = [
            "You are a national policy agent. Decide next-round measures.",
            "Return the policy JSON with keys {type, actor, import_tariff, export_quota, import_multiplier, rationale?, plan?} following the format below.",
            "Constraints: indices in [0,{N-1}], import_tariff must be one of [0.0, 0.1, 0.2, 0.3], export_quota must be one of [1.0, 0.9, 0.8], import_multiplier in [0.5,2.0]; per-type <= {K} sectors.",
            "Guidance: adjust import_tariff and export_quota in a balanced manner to manage domestic prices and external trade exposure.",
            "Keep import_multiplier empty {}; do not set it.",
            "Objective: Maximize your own reward r = w_income*income_growth_last - w_price*price_mean_last + w_trade*trade_balance_last.",
            f"Current reward weights: w_income={w_income}, w_price={w_price}, w_trade={w_trade}.",
            "Higher reward is better for you; choose actions to increase it next evaluation.",
        ]
        if self.use_json_block:
            lines.extend(
                [
                    "Output format:",
                    "<think>brief reasoning (optional)</think>",
                    "<json>{...policy JSON...}</json>",
                    "Do NOT include code fences/backticks. Use double quotes for all keys/strings; no single quotes.",
                    "All reasoning must stay inside <think>; the <json> block must contain only the policy object.",
                    "Exactly one <json> block, with no extra text before/after.",
                    "The <json> block MUST contain a RAW JSON object (not a quoted/escaped JSON string). Do not surround it with quotes or add backslashes.",
                ]
            )
            # 单次示例：提醒模型避免输出被引号包裹或转义的 JSON
            good_example = (
                "<think>brief reasoning (optional)</think>\n"
                + "<json>{\"type\":\"policy_bundle\",\"actor\":\""
                + actor
                + "\",\"import_tariff\":{\"0\":0.05},\"export_quota\":{},\"import_multiplier\":{},\"rationale\":\"...\",\"plan\":\"...\"}</json>"
            )
            bad_example = (
                "<json>\"{\\\"type\\\":\\\"policy_bundle\\\",\\\"actor\\\":\\\""
                + actor
                + "\\\",\\\"import_tariff\\\":{\\\"0\\\":0.05},\\\"export_quota\\\":{},\\\"import_multiplier\\\":{}}\"</json>"
            )
            lines.extend(
                [
                    "Good example (RAW JSON object):",
                    good_example,
                    "Bad example (INVALID — quoted/escaped JSON string; do not do this):",
                    bad_example,
                ]
            )
        else:
            lines.append("Reply with ONLY the policy JSON object (no commentary).")

        lines.extend(
            [
                f"CURRENT_ACTOR={actor}. Your JSON must set actor to '{actor}' exactly.",
                "Your JSON must set type to 'policy_bundle' exactly.",
                "Always include keys import_tariff, export_quota, import_multiplier. Use empty objects {} if no changes.",
                "For each of import_tariff/export_quota/import_multiplier: it MUST be a FLAT JSON object mapping sector indices (integers 0..{N-1}) to numeric values.",
                "Keys MUST be integer or stringified integer (e.g. 0 or \"0\"). Do NOT use nested objects or keys like 'sector'/'value'.",
                "No extra keys are allowed under these maps.",
                "Example JSON (edit values as needed):",
            ]
        )
        example = (
            "<json>{\"type\":\"policy_bundle\",\"actor\":\""
            + actor
            + "\",\"import_tariff\":{\"0\":0.05},\"export_quota\":{\"2\":0.9},\"import_multiplier\":{},\"rationale\":\"...\",\"plan\":\"...\"}</json>"
            if self.use_json_block
            else "{\"type\":\"policy_bundle\",\"actor\":\""
            + actor
            + "\",\"import_tariff\":{\"0\":0.05},\"export_quota\":{\"2\":0.9},\"import_multiplier\":{},\"rationale\":\"...\",\"plan\":\"...\"}"
        )
        lines.append(example)
        lines.extend(
            [
                f"Sectors: 0..{self.n-1}; per_type_limit={self.maxn}",
                "Context metrics:",
                f"income={metrics.get('income_last')}, price_mean={metrics.get('price_mean_last')}, trade_balance={metrics.get('trade_balance_last')}",
                f"delta: income_d={delta.get('income_d')}, price_mean_d={delta.get('price_mean_d')}, trade_balance_d={delta.get('trade_balance_d')}",
                f"prev_action={prev}",
                f"last_actions_all_countries={last_actions}",
                f"price_topk_d={vectors_summary.get('price_topk_d')}",
                f"output_topk_d={vectors_summary.get('output_topk_d')}",
            ]
        )
        limit_str = "None" if self.maxn is None else str(self.maxn)
        return "\n".join(lines).replace("{N-1}", str(self.n - 1)).replace("{K}", limit_str)

    def _chat_payload(self, prompt: str) -> Dict[str, Any]:
        if self.use_json_block:
            system_msg = (
                "Provide reasoning inside <think>...</think> (optional) and the final policy JSON inside "
                "<json>{...}</json>. The JSON must be valid, use double-quoted keys/strings, and include no extra text. "
                "The <json> block must contain a RAW JSON object (not a quoted or escaped JSON string)."
            )
        else:
            system_msg = "Reply ONLY with strict JSON object. No commentary."
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        # 尝试使用 Function Calling 约束输出为结构化 JSON（多数 vLLM + Qwen 支持）
        if self.use_tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_policy",
                        "description": "Return the next-round policy bundle as a JSON object.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["policy_bundle"]},
                                "actor": {"type": "string"},
                                "import_tariff": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                },
                                "export_quota": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                },
                                "import_multiplier": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                                },
                                "rationale": {"type": "string"},
                                "plan": {"type": "string"},
                            },
                            "required": [
                                "type",
                                "actor",
                                "import_tariff",
                                "export_quota",
                                "import_multiplier",
                            ],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "function", "function": {"name": "submit_policy"}}
        return payload

    def _post(self, payload: Dict[str, Any]) -> str:
        url = urljoin(self.base_url, "chat/completions")
        body = json.dumps(payload).encode("utf-8")
        req = _urlrequest.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with _urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except _urlerror.HTTPError as err:
            raw = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {err.code}: {raw[:200]}")
        except _urlerror.URLError as err:
            raise RuntimeError(f"LLM URL error: {err}")

    def _to_action_strict(self, data: Dict[str, Any], actor: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("LLM content must be JSON object")
        action = dict(data)
        action.setdefault("actor", actor)
        if action.get("actor") != actor:
            raise ValueError("actor mismatch with current country")
        action = _validate_action_strict(action, n_sectors=self.n, max_per_type=self.maxn)
        return action

    def _extract_json_block(self, content: str) -> str:
        text = content.strip()
        if self.use_json_block:
            match = re.search(r"<json>\s*(\{[\s\S]*\})\s*</json>", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        text_no_think = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text_no_think, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(\{[\s\S]*?\})\s*```", text_no_think)
        if match:
            return match.group(1).strip()
        start = text_no_think.find("{")
        end = text_no_think.rfind("}")
        if 0 <= start < end:
            return text_no_think[start : end + 1]
        raise ValueError("missing <json> block with policy object")

    def _parse_json_strictish(self, content: str) -> Dict[str, Any]:
        queue: List[str] = [content.strip()]
        seen: set[str] = set()

        def _apply_arithmetic(expr: str) -> tuple[str, bool]:
            def repl(match: re.Match[str]) -> str:
                left = float(match.group(1))
                op = match.group(2)
                right = float(match.group(3))
                if op == "+":
                    val = left + right
                elif op == "-":
                    val = left - right
                elif op == "*":
                    val = left * right
                else:
                    if abs(right) < 1e-12:
                        raise ValueError("division by zero in expression")
                    val = left / right
                return f": {val:.6f}{match.group(4)}"

            pattern = re.compile(r":\s*(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)(\s*[}\],])")
            changed = False

            def safe(match: re.Match[str]) -> str:
                nonlocal changed
                changed = True
                return repl(match)

            new_expr = pattern.sub(safe, expr)
            return new_expr, changed

        while queue:
            current = queue.pop(0).strip()
            if not current or current in seen:
                continue
            seen.add(current)
            # 首轮解析：直接尝试把字符串视作 JSON
            try:
                obj = json.loads(current)
                # 若已经得到字典对象，直接返回
                if isinstance(obj, dict):
                    return obj
                # 某些模型会返回再次编码成字符串的 JSON，此处追加一轮解析
                if isinstance(obj, str):
                    s = obj.strip()
                    if s.startswith("{") and s.endswith("}"):
                        queue.append(s)
                        continue
            except Exception:
                pass

            # 启发式：字面看像对象但含大量转义引号（例如 {\"key\":...}）
            # 处理方式：再包一层 JSON 字符串，解码一次还原，再继续解析
            text = current
            looks_like_object = text.startswith("{") and text.endswith("}")
            if looks_like_object and "\\\"" in text:
                try:
                    wrapped = '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
                    decoded = json.loads(wrapped)  # 转成 {"key":...} 这样的普通字符串
                    if isinstance(decoded, str) and decoded.strip().startswith("{") and decoded.strip().endswith("}"):
                        queue.append(decoded.strip())
                        continue
                except Exception:
                    pass
                # 兜底：直接去除常见转义，尽力修复
                de = text.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                if de != text:
                    queue.append(de)

            # 启发式：模型把顶层对象拆成 `...}, {"export_quota":...`
            # 处理方式：把 `} , { "key"` 合并成 `, "key"`
            if looks_like_object and re.search(r"}\s*,\s*{\s*\"(export_quota|import_multiplier)\"", text):
                merged = re.sub(r"}\s*,\s*{\s*\"", ', "', text)
                if merged != text:
                    queue.append(merged)

            # 启发式：删除破坏 JSON 的空键（\"\": value）
            if looks_like_object and '""' in text:
                no_empty_key = re.sub(r",\s*\"\"\s*:\s*[^,}\]]*", '', text)
                no_empty_key = re.sub(r"\{\s*\"\"\s*:\s*[^,}\]]*,?\s*", '{', no_empty_key)
                if no_empty_key != text:
                    queue.append(no_empty_key)
            quoted_numeric = re.sub(r"({|,)\s*(\d+)\s*:", lambda m: f'{m.group(1)} "{m.group(2)}":', current)
            if quoted_numeric != current:
                queue.append(quoted_numeric)
            if '"' not in current and "'" in current:
                queue.append(current.replace("'", '"'))
            trimmed = re.sub(r",\s*([}\]])", r"\1", current)
            if trimmed != current:
                queue.append(trimmed)
            arithmetic_fixed, changed = _apply_arithmetic(current)
            if changed:
                queue.append(arithmetic_fixed)
        raise ValueError("LLM returned non-JSON content")

    def _build_ultrathink_feedback(
        self,
        err: Exception,
        actor: str,
        content_head: str,
        json_head: str,
        n_sectors: int,
        maxn: Optional[int],
    ) -> str:
        message = str(err)
        lines: List[str] = []
        if self.use_json_block:
            lines.append(
                "Your previous reply was rejected. Output format must be <think>...</think> optionally followed by "
                "<json>{...}</json> containing ONLY the policy JSON."
            )
        else:
            lines.append("Your previous JSON was rejected. Reply with ONLY the policy JSON object.")

        limits = {
            "import_tariff": "[-1, 1]",
            "export_quota": "[0, 1]",
            "import_multiplier": "keep empty {}",
        }
        match = re.search(r"(import_tariff|export_quota|import_multiplier) value out of range: ([^\s]+)", message)
        if match:
            name, val = match.group(1), match.group(2)
            rng = limits.get(name, "")
            lines.append(f"Error: {name} value out of range ({val}). Valid range is {rng}. Adjust values accordingly.")
        match = re.search(r"(import_tariff|export_quota|import_multiplier) sector out of range: (-?\d+)", message)
        if match:
            name, idx = match.group(1), match.group(2)
            lines.append(f"Error: {name} sector index {idx} is invalid. Use integers in [0,{n_sectors-1}] only; remove invalid keys.")
        match = re.search(r"too many sectors: (\d+) > (\d+)", message)
        if match:
            got, mx = match.group(1), match.group(2)
            lines.append(f"Error: too many sectors per type ({got}>{mx}). Reduce to at most {mx} sectors in each map.")
        if "must be object map" in message:
            lines.append("Error: Each of import_tariff/export_quota/import_multiplier must be a FLAT JSON object mapping sector indices to numbers.")
        if "actor mismatch" in message:
            lines.append(f"Error: actor mismatch. Set 'actor' to '{actor}' exactly.")
        if "missing <json> block" in message:
            lines.append("Error: Missing <json>...</json> block. Place the policy object inside <json> tags exactly once.")
        if "LLM content must be JSON object" in message or isinstance(err, json.JSONDecodeError):
            lines.append("Error: invalid JSON. Use double quotes for all keys/strings; no backticks or code fences; keep text outside the <json> block empty.")
        if re.search(r"\d+\s*[-+*/]\s*\d+", json_head):
            lines.append("Error: do not include arithmetic expressions like 0.875-0.05 inside JSON values; compute the numeric result first.")
        # JSON block 模式的额外提示
        if ('\\\"' in (json_head or "")) or re.search(r"^\s*\"\{", (content_head or "")):
            lines.append("Error: do not return a quoted/escaped JSON string inside <json>. Put a RAW JSON object there, e.g., {\"type\":\"policy_bundle\",...} without extra escaping.")
        per_type = "unlimited" if maxn is None else str(maxn)
        lines.append(
            f"Constraints recap: actor='{actor}', type='policy_bundle'; import_tariff in [-1,1]; export_quota in [0,1]; import_multiplier={{}}; indices 0..{n_sectors-1}; per-type <= {per_type} sectors."
        )
        return "\n".join(lines)

    def _write_log(self, actor: str, t: int, record: Dict[str, Any]) -> None:
        directory = self._log_base / self._run_tag
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{actor}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            entry = {"t": int(t), **record}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_policy_from_spec(spec: Optional[str], n_sectors: int) -> PolicyFn:
    """策略工厂：生产环境默认支持 llm:* 规格，也可启用 search:* 搜索策略。"""
    if not spec:
        raise ValueError("policy spec must be provided (e.g., llm:...)")

    def _parse_float_list(raw: str) -> List[float]:
        out: List[float] = []
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(float(token))
            except Exception:
                continue
        return out

    if spec.startswith("llm"):
        model = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        log_dir = os.getenv("AGENT_LOG_DIR", None)
        rest = spec[3:].lstrip(": ") if len(spec) > 3 else ""
        if rest:
            if "=" not in rest:
                model = rest.strip()
            else:
                for part in rest.split(","):
                    if not part or "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "model":
                        model = value
                    elif key == "base":
                        base = value
                    elif key in ("key", "api_key"):
                        api_key = value
                    elif key in ("temp", "temperature"):
                        try:
                            temp = float(value)
                        except Exception:
                            pass
                    elif key in ("log_dir", "logdir", "log-directory"):
                        log_dir = value
        return LLMPolicyAdapter(
            n_sectors=n_sectors,
            model=model,
            base_url=base,
            api_key=api_key,
            temperature=temp,
            log_dir=log_dir,
        )
    if spec.startswith("search"):
        from .search import SearchPolicyAdapter

        rest = spec[6:].lstrip(": ") if len(spec) > 6 else ""
        lookahead_rounds = 1
        lookahead_steps: Optional[int] = None
        tariff_grid: Optional[List[float]] = None
        quota_grid: Optional[List[float]] = None
        target_sectors: Optional[List[int]] = None
        objective = "reward"
        method = "grid"

        if rest:
            for part in rest.split(","):
                if not part or "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ("lookahead", "rounds"):
                    try:
                        lookahead_rounds = int(value)
                    except Exception:
                        continue
                elif key in ("steps", "lookahead_steps"):
                    try:
                        lookahead_steps = int(value)
                    except Exception:
                        continue
                elif key in ("tariffs", "tariff_grid"):
                    parsed = _parse_float_list(value)
                    if parsed:
                        tariff_grid = parsed
                elif key in ("quotas", "quota_grid"):
                    parsed = _parse_float_list(value)
                    if parsed:
                        quota_grid = parsed
                elif key == "objective":
                    objective = value or objective
                elif key == "method":
                    method = "grid"  # 仅保留 grid，忽略其他取值以免混淆
                elif key == "sectors":
                    # 解析 static targeting 列表；当为空时保留为 None 以回退默认选择
                    val = value.strip()
                    if not val:
                        target_sectors = None
                    else:
                        try:
                            sep = ";" if ";" in val else ","
                            parsed = [int(x) for x in val.split(sep) if x.strip()]
                            target_sectors = parsed if parsed else None
                        except Exception:
                            warnings.warn(f"Failed to parse sectors: '{val}'. Falling back to default.", UserWarning)
                            # 解析失败不改变现有值（默认为 None）
                            target_sectors = target_sectors

        if target_sectors is None:
            raise ValueError("search policy requires sectors=... to specify target sectors")
        return SearchPolicyAdapter(
            n_sectors=n_sectors,
            lookahead_rounds=int(lookahead_rounds),
            lookahead_steps=int(lookahead_steps) if lookahead_steps is not None else None,
            tariff_grid=tariff_grid,
            quota_grid=quota_grid,
            objective=objective,
            target_sectors=target_sectors,
            method="grid",
        )
    raise ValueError(f"unknown policy spec: {spec}")


__all__ = ["PolicyAdapter", "LLMPolicyAdapter", "build_policy_from_spec"]
