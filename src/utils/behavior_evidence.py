"""
异常行为抓拍存证：睡觉（超时或连续帧）、使用手机（即时 + 冷却）
图片写入 CAPTURES_DIR，元数据写入 SQLite behavior_evidence 表。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, CAPTURES_DIR, CHEATING_CONFIG, EVIDENCE_CONFIG
from utils.database import Database


class BehaviorEvidenceCollector:
    """按 track 维护睡觉时长/连帧计数；玩手机按冷却抓拍。"""

    def __init__(self, db_path: str, captures_dir: Optional[str] = None) -> None:
        self.db = Database(db_path)
        self.captures_dir = Path(captures_dir or CAPTURES_DIR)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self._min_conf = float(
            EVIDENCE_CONFIG.get("min_confidence", CHEATING_CONFIG["confidence_threshold"])
        )
        self._sleep_sec = float(EVIDENCE_CONFIG["sleep_duration_sec"])
        self._sleep_consec = int(EVIDENCE_CONFIG["sleep_consecutive_frames"])
        self._phone_cd = float(EVIDENCE_CONFIG["phone_cooldown_sec"])

        self._sleep_start: Dict[int, float] = {}
        self._sleep_consec_count: Dict[int, int] = {}
        self._sleep_episode_saved: Dict[int, bool] = {}
        self._phone_last: Dict[int, float] = {}

    def prune_tracks(self, active_ids: List[int]) -> None:
        """跟踪丢失时清理状态，避免内存增长。"""
        active = set(active_ids)
        for tid in list(self._sleep_start.keys()):
            if tid not in active:
                self._end_sleep_episode(tid)
        for tid in list(self._phone_last.keys()):
            if tid not in active:
                self._phone_last.pop(tid, None)

    def _end_sleep_episode(self, track_id: int) -> None:
        self._sleep_start.pop(track_id, None)
        self._sleep_consec_count.pop(track_id, None)
        self._sleep_episode_saved.pop(track_id, None)

    def _save_image(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        tag: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """返回 (绝对路径, 相对项目根路径)。"""
        ts = int(time.time() * 1000)
        fname = f"{tag}_tid{track_id}_{ts}.jpg"
        path = self.captures_dir / fname
        try:
            vis = frame_bgr.copy()
            x1, y1, x2, y2 = map(int, bbox)
            h, w = vis.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{tag} tid={track_id}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(path), vis)
            rel = os.path.relpath(str(path), BASE_DIR).replace("\\", "/")
            return str(path), rel
        except Exception as e:
            print(f"抓拍保存失败: {e}")
            return None, None

    def process_track(
        self,
        track_id: int,
        action_name: str,
        confidence: float,
        frame_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> List[Dict[str, Any]]:
        """单帧、单目标。返回本帧产生的存证事件（0 或 1 条）。"""
        events: List[Dict[str, Any]] = []

        if confidence < self._min_conf:
            if action_name == "sleep" or track_id in self._sleep_start:
                self._end_sleep_episode(track_id)
            return events

        if action_name == "using_phone":
            self._end_sleep_episode(track_id)
            now = time.time()
            last = self._phone_last.get(track_id, 0.0)
            if now - last < self._phone_cd:
                return events
            abs_path, rel = self._save_image(frame_bgr, bbox, track_id, "phone")
            if not abs_path or not rel:
                return events
            self._phone_last[track_id] = now
            detail = {"cooldown_sec": self._phone_cd}
            rid = self.db.add_behavior_evidence(
                "using_phone",
                track_id,
                rel,
                float(confidence),
                "phone_instant",
                json.dumps(detail, ensure_ascii=False),
            )
            events.append(
                {
                    "type": "using_phone",
                    "track_id": track_id,
                    "path": abs_path,
                    "db_id": rid,
                    "reason": "phone_instant",
                }
            )
            return events

        if action_name == "sleep":
            tnow = time.time()
            if track_id not in self._sleep_start:
                self._sleep_start[track_id] = tnow
                self._sleep_consec_count[track_id] = 0
            self._sleep_consec_count[track_id] = self._sleep_consec_count.get(track_id, 0) + 1

            duration = tnow - self._sleep_start[track_id]
            consec = self._sleep_consec_count[track_id]
            saved = self._sleep_episode_saved.get(track_id, False)

            trigger: Optional[str] = None
            if not saved:
                if duration >= self._sleep_sec:
                    trigger = "duration_30s"
                elif consec >= self._sleep_consec:
                    trigger = "consecutive_frames"

            if trigger:
                abs_path, rel = self._save_image(frame_bgr, bbox, track_id, "sleep")
                if abs_path and rel:
                    self._sleep_episode_saved[track_id] = True
                    detail = {
                        "duration_sec": round(duration, 2),
                        "consecutive_frames": consec,
                        "threshold_sec": self._sleep_sec,
                        "threshold_frames": self._sleep_consec,
                    }
                    rid = self.db.add_behavior_evidence(
                        "sleep",
                        track_id,
                        rel,
                        float(confidence),
                        trigger,
                        json.dumps(detail, ensure_ascii=False),
                    )
                    events.append(
                        {
                            "type": "sleep",
                            "track_id": track_id,
                            "path": abs_path,
                            "db_id": rid,
                            "reason": trigger,
                        }
                    )
            return events

        self._end_sleep_episode(track_id)
        return events
