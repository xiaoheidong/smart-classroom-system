"""
飞书自定义机器人 Webhook（文本消息，支持「加签」安全设置）
文档: https://open.feishu.cn/document/client-docs/bot-v2/add-custom-bot
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _gen_sign(secret: str) -> Tuple[str, str]:
    """
    飞书文档：将 timestamp + \"\\n\" + 密钥 作为 HMAC-SHA256 的 **密钥**，
    对**空内容**计算签名后再 Base64（与 Java 示例 SecretKeySpec(stringToSign) + doFinal(空) 一致）。
    文档: https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN
    """
    timestamp = str(int(time.time()))
    string_to_sign = f"{timestamp}\n{secret}"
    sign = base64.b64encode(
        hmac.new(
            string_to_sign.encode("utf-8"),
            b"",
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    return timestamp, sign


def send_feishu_text(
    webhook_url: str,
    text: str,
    secret: Optional[str] = None,
    timeout: float = 15.0,
) -> Tuple[bool, str]:
    """
    发送纯文本。若配置了 secret（机器人安全设置-加签），自动附带 timestamp、sign。
    """
    url = (webhook_url or "").strip()
    if not url:
        return False, "未配置 Webhook URL"

    body: Dict[str, Any] = {"msg_type": "text", "content": {"text": text}}
    sec = (secret or "").strip()
    if sec:
        ts, sign = _gen_sign(sec)
        body["timestamp"] = ts
        body["sign"] = sign

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return False, f"HTTP {e.code}: {err_body}"
    except urllib.error.URLError as e:
        return False, f"网络错误: {e.reason}"
    except Exception as e:
        return False, str(e)

    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        return True, raw

    code = j.get("code")
    if code == 0:
        return True, "ok"
    return False, j.get("msg", raw)


def send_session_attendance_to_feishu(
    db_path: str,
    session_id: int,
    webhook_url: str,
    secret: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    根据场次 ID 汇总本场时间、应到/实到/未到，并推送到飞书。
    """
    from utils.database import Database

    url = (webhook_url or "").strip()
    if not url:
        return True, "已跳过（未配置 FEISHU_WEBHOOK_URL）"

    db = Database(db_path)
    meta = db.get_session_row(session_id)
    if not meta:
        return False, f"场次 #{session_id} 不存在"

    present_records, absent = db.get_session_summary(session_id)
    present_n = len(present_records)
    absent_n = len(absent)
    total_n = present_n + absent_n

    started = meta["started_at"] or ""
    ended = meta["ended_at"] or "（未结束）"

    lines = [
        f"【智慧教室·课堂签到】场次 #{session_id}",
        f"开始时间：{started}",
        f"结束时间：{ended}",
        "",
        f"应到 {total_n} 人，实到 {present_n} 人，未到 {absent_n} 人",
        "",
    ]
    if absent_n == 0:
        lines.append("未到名单：无")
    else:
        lines.append("未到名单：")
        for a in absent:
            lines.append(f"· {a['name']}（{a['id']}）")

    text = "\n".join(lines)
    return send_feishu_text(url, text, secret=secret)
