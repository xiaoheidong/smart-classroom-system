"""
DeepSeek Chat API（OpenAI 兼容接口），仅 HTTPS + 标准库，无额外依赖。
文档: https://api-docs.deepseek.com
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


def chat_completion(
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = 120.0,
    temperature: float = 0.4,
) -> Tuple[bool, str]:
    """
    调用 chat/completions。
    Returns:
        (True, 助手正文) 或 (False, 错误说明)
    """
    key = (api_key or "").strip()
    if not key:
        return False, "未配置 DEEPSEEK_API_KEY 环境变量。"

    url = f"{base_url.rstrip('/')}/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {key}")

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            j = json.loads(err_body)
            msg = j.get("error", {})
            if isinstance(msg, dict):
                return False, msg.get("message", err_body)[:2000]
        except json.JSONDecodeError:
            pass
        return False, f"HTTP {e.code}: {err_body[:1500]}"
    except urllib.error.URLError as e:
        return False, f"网络错误: {e.reason}"
    except Exception as e:
        return False, str(e)

    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        return False, "响应非 JSON"

    if "choices" in j and j["choices"]:
        ch0 = j["choices"][0]
        msg = ch0.get("message") or {}
        content = msg.get("content", "")
        if content:
            return True, content.strip()
        return False, "响应中无正文"

    err = j.get("error")
    if isinstance(err, dict):
        return False, err.get("message", str(j))[:2000]
    return False, str(j)[:2000]
