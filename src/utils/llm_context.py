"""
为「智能问答」拼装当日结构化上下文（考勤 + 异常抓拍），供 DeepSeek 仅基于事实回答。
"""
from __future__ import annotations

import os
from datetime import date
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from utils.database import Database

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR


def build_daily_context(db: "Database", day: Optional[date] = None, max_evidence_lines: int = 80) -> str:
    """生成可放入 system 或 user 附带的纯文本事实块。"""
    d = day or date.today()
    lines: List[str] = []
    lines.append(f"【数据日期】{d.isoformat()}（以下均为本机 SQLite 查询结果，非虚构）")
    lines.append("")

    present, absent = [], []
    if d == date.today():
        try:
            present, absent = db.get_today_attendance_summary()
        except Exception as e:
            lines.append(f"【考勤】读取失败: {e}")
    else:
        lines.append("【考勤】未附带历史日期的签到汇总（当前仅支持「今天」的考勤摘要）。")
        lines.append("")

    if d == date.today():
        lines.append(
            f"【今日考勤摘要】已出现在当日签到记录中的学生约 {len(present)} 人；"
            f"当日尚无签到记录的学生约 {len(absent)} 人（以数据库为准）。"
        )
        if present:
            lines.append("已签到姓名(节选): " + "、".join(present[:40]) + ("…" if len(present) > 40 else ""))
        if absent:
            lines.append("当日无签到记录的学生姓名(节选): " + "、".join(absent[:40]) + ("…" if len(absent) > 40 else ""))
        lines.append("")

    ev = []
    try:
        ev = db.list_behavior_evidence_for_date(d, limit=500)
    except Exception as e:
        lines.append(f"【异常抓拍】读取失败: {e}")
        return "\n".join(lines)

    sleep_n = sum(1 for x in ev if (x.get("behavior_type") == "sleep"))
    phone_n = sum(1 for x in ev if (x.get("behavior_type") == "using_phone"))
    lines.append(
        f"【异常行为抓拍】{d.isoformat()} 共 {len(ev)} 条（睡觉类 {sleep_n} 条，使用手机类 {phone_n} 条）。"
    )
    lines.append("说明：图片路径为相对项目根目录；文件是否仍存在以磁盘为准。")
    lines.append("")

    for i, e in enumerate(ev[:max_evidence_lines], 1):
        rel = e.get("image_path", "")
        abs_hint = os.path.normpath(os.path.join(BASE_DIR, rel)) if rel else ""
        lines.append(
            f"{i}. 类型={e.get('behavior_type')} "
            f"track_id={e.get('track_id')} "
            f"时间={e.get('created_at')} "
            f"置信度={float(e.get('confidence') or 0):.3f} "
            f"触发={e.get('trigger_reason')} "
            f"图片相对路径={rel} "
            f"本地路径={abs_hint}"
        )
    if len(ev) > max_evidence_lines:
        lines.append(
            f"… 共 {len(ev)} 条抓拍，以上仅列前 {max_evidence_lines} 条。"
        )

    return "\n".join(lines)
