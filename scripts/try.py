import requests
import json

# ===== Prompt from agent logs (H.jsonl) =====
SYSTEM_MSG = (
    "Provide reasoning inside <think>...</think> (optional) and the final policy JSON inside "
    "<json>{...}</json>. The JSON must be valid, use double-quoted keys/strings, and include no extra text. "
    "The <json> block must contain a RAW JSON object (not a quoted or escaped JSON string)."
)

PROMPT = """
You are a national policy agent. Decide next-round measures.
Return the policy JSON with keys {type, actor, import_tariff, export_quota, import_multiplier, rationale?, plan?} following the format below.
Constraints: indices in [0,5], import_tariff in [-1,1], export_quota in [0,1], import_multiplier in [0.5,2.0]; per-type <= None sectors.
Guidance: adjust import_tariff and export_quota in a balanced manner to manage domestic prices and external trade exposure.
Keep import_multiplier empty {}; do not set it.
Objective: Maximize your own reward r = w_income*income_growth_last - w_price*price_mean_last + w_trade*trade_balance_last.
Current reward weights: w_income=1.0, w_price=0.5, w_trade=0.5.
Higher reward is better for you; choose actions to increase it next evaluation.

Output format:
<think>brief reasoning (optional)</think>
<json>{...policy JSON...}</json>
Do NOT include code fences/backticks. Use double quotes for all keys/strings; no single quotes.
All reasoning must stay inside <think>; the <json> block must contain only the policy object.
Exactly one <json> block, with no extra text before/after.
The <json> block MUST contain a RAW JSON object (not a quoted/escaped JSON string). Do not surround it with quotes or add backslashes.

Good example (RAW JSON object):
<think>brief reasoning (optional)</think>
<json>{"type":"policy_bundle","actor":"H","import_tariff":{"0":0.05},"export_quota":{},"import_multiplier":{},"rationale":"...","plan":"..."}</json>

Bad example (INVALID — quoted/escaped JSON string; do not do this):
<json>"{\"type\":\"policy_bundle\",\"actor\":\"H\",\"import_tariff\":{\"0\":0.05},\"export_quota\":{},\"import_multiplier\":{}}"</json>

CURRENT_ACTOR=H. Your JSON must set actor to 'H' exactly.
Your JSON must set type to 'policy_bundle' exactly.
Always include keys import_tariff, export_quota, import_multiplier. Use empty objects {} if no changes.
For each of import_tariff/export_quota/import_multiplier: it MUST be a FLAT JSON object mapping sector indices (integers 0..5) to numeric values.
Keys MUST be integer or stringified integer (e.g. 0 or "0"). Do NOT use nested objects or keys like 'sector'/'value'.
No extra keys are allowed under these maps.
Example JSON (edit values as needed):
<json>{"type":"policy_bundle","actor":"H","import_tariff":{"0":0.05},"export_quota":{"2":0.9},"import_multiplier":{},"rationale":"...","plan":"..."}</json>

Sectors: 0..5; per_type_limit=None
"""

# 直接在此处配置，不使用环境变量
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = "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"  # 示例：sk-or-v1-xxxx
MODEL = "minimax/minimax-m2:free"
HTTP_REFERER = "<YOUR_SITE_URL>"  # 可选：用于 openrouter.ai 排名
X_TITLE = "<YOUR_SITE_NAME>"      # 可选：用于 openrouter.ai 排名

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
if HTTP_REFERER:
    headers["HTTP-Referer"] = HTTP_REFERER
if X_TITLE:
    headers["X-Title"] = X_TITLE

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": PROMPT},
    ],
    # 可选：部分模型支持强制 JSON 输出
    # "response_format": {"type": "json_object"},
}

def _extract_json_block(text: str) -> str:
    s = (text or "").strip()
    # prefer <json> ... </json>
    import re
    m = re.search(r"<json>\s*(\{[\s\S]*\})\s*</json>", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # drop <think> ... </think>
    s2 = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    # fenced ```json
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s2, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(\{[\s\S]*?\})\s*```", s2)
    if m:
        return m.group(1).strip()
    # first {...}
    a = s2.find("{")
    b = s2.rfind("}")
    if 0 <= a < b:
        return s2[a : b + 1]
    raise ValueError("missing <json> block with policy object")


def _parse_json_strictish(content: str):
    import re, json as _json
    queue = [content.strip()]
    seen = set()

    def _apply_arith(expr: str):
        pat = re.compile(r":\s*(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)(\s*[}\],])")
        changed = False

        def repl(m: re.Match[str]) -> str:
            nonlocal changed
            left = float(m.group(1)); op = m.group(2); right = float(m.group(3))
            if op == "+": val = left + right
            elif op == "-": val = left - right
            elif op == "*": val = left * right
            else:
                val = left / right if abs(right) > 1e-12 else float("inf")
            changed = True
            return f": {val:.6f}{m.group(4)}"

        return pat.sub(repl, expr), changed

    while queue:
        cur = queue.pop(0).strip()
        if not cur or cur in seen:
            continue
        seen.add(cur)
        try:
            obj = _json.loads(cur)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                s = obj.strip()
                if s.startswith("{") and s.endswith("}"):
                    queue.append(s)
                    continue
        except Exception:
            pass

        looks_like_obj = cur.startswith("{") and cur.endswith("}")
        if looks_like_obj and '\\"' in cur:
            try:
                wrapped = '"' + cur.replace("\\", "\\\\").replace('"', '\\"') + '"'
                decoded = _json.loads(wrapped)
                if isinstance(decoded, str) and decoded.strip().startswith("{") and decoded.strip().endswith("}"):
                    s = decoded.strip()
                    queue.append(s)
                    queue.append(s.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' '))
                    continue
            except Exception:
                pass
            de = cur.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
            if de != cur:
                queue.append(de)

        if looks_like_obj:
            # merge split top-level maps: '}, {"export_quota"' -> ', "export_quota"'
            import re as _re
            if _re.search(r"}\s*,\s*{\s*\"(export_quota|import_multiplier)\"", cur):
                merged = _re.sub(r"}\s*,\s*{\s*\"", ', "', cur)
                if merged != cur:
                    queue.append(merged)
            if '""' in cur:
                no_empty = _re.sub(r",\s*\"\"\s*:\s*[^,}\]]*", '', cur)
                no_empty = _re.sub(r"\{\s*\"\"\s*:\s*[^,}\]]*,?\s*", '{', no_empty)
                if no_empty != cur:
                    queue.append(no_empty)

        quoted_numeric = re.sub(r"({|,)\s*(\d+)\s*:", lambda m: f'{m.group(1)} "{m.group(2)}":', cur)
        if quoted_numeric != cur:
            queue.append(quoted_numeric)
        if '"' not in cur and "'" in cur:
            queue.append(cur.replace("'", '"'))
        trimmed = re.sub(r",\s*([}\]])", r"\1", cur)
        if trimmed != cur:
            queue.append(trimmed)
        ar, ch = _apply_arith(cur)
        if ch:
            queue.append(ar)
    raise ValueError("LLM returned non-JSON content")


def run() -> None:
    result = {
        "http_status": None,
        "elapsed_s": None,
        "model": MODEL,
        "raw_content_head": None,
        "tool_args_head": None,
        "parsed": None,
        "ok": False,
        "error": None,
    }
    try:
        resp = requests.post(
            url=f"{BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        elapsed = getattr(resp, "elapsed", None)
        elapsed_s = getattr(elapsed, "total_seconds", lambda: None)()
        result["http_status"] = resp.status_code
        result["elapsed_s"] = elapsed_s
        print(f"HTTP {resp.status_code}, elapsed={elapsed_s if elapsed_s is not None else 'n/a'}s")
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            head = resp.text[:2000]
            print("[raw] response head:\n" + head)
            result["raw_content_head"] = head
            raise

        # pretty print head for quick debug
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        print("=== JSON (head) ===")
        print(pretty[:5000])

        # Extract message content / tools / function_call
        choices = data.get("choices", []) or []
        msg = (choices[0] or {}).get("message", {}) if choices else {}
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        func_call = msg.get("function_call") or {}

        if tool_calls:
            fn = (tool_calls[0] or {}).get("function", {}) or {}
            args = fn.get("arguments", "") or ""
            result["tool_args_head"] = args[:500]
            data_obj = _parse_json_strictish(args)
        elif func_call and isinstance(func_call.get("arguments"), str):
            args = func_call.get("arguments") or ""
            result["tool_args_head"] = args[:500]
            data_obj = _parse_json_strictish(args)
        else:
            if not isinstance(content, str):
                raise ValueError("LLM returned empty content")
            result["raw_content_head"] = content[:500]
            json_payload = _extract_json_block(content)
            data_obj = _parse_json_strictish(json_payload)

        result["parsed"] = data_obj
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
        print("[error]", e)
    finally:
        try:
            with open("try-out", "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
            print("[saved] try-out")
        except Exception as fe:
            print("[warn] failed to save try-out:", fe)


if __name__ == "__main__":
    run()
