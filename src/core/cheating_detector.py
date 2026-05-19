"""
课堂异常行为检测模块
基于行为分类结果进行规则判断与时序平滑
"""
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTION_DISPLAY_NAMES, ALERT_ACTIONS, CHEATING_CONFIG, NORMAL_ACTIONS


class CheatingDetector:
    """课堂异常行为检测器。"""
    
    def __init__(self, confidence_threshold: float = CHEATING_CONFIG['confidence_threshold']):
        """初始化检测器。"""
        self.confidence_threshold = confidence_threshold

        self.action_history = defaultdict(lambda: deque(maxlen=30))
        self.alerts = defaultdict(list)
        self.stats = {
            'total': 0,
            'normal': 0,
            'cheating': 0,
            'by_type': {
                '睡觉': 0,
                '使用手机': 0
            }
        }

    def detect_cheating(
        self,
        action_name: str,
        confidence: float,
        track_id: Optional[int] = None,
    ) -> Dict:
        """根据行为分类结果判断是否异常。"""
        result = {
            'is_cheating': False,
            'cheating_type': '正常',
            'confidence': confidence,
            'details': {
                'action_name': action_name,
                'display_name': ACTION_DISPLAY_NAMES.get(action_name, action_name),
            }
        }

        if confidence < self.confidence_threshold:
            result['details']['reason'] = '分类置信度不足'
        elif action_name in ALERT_ACTIONS:
            result['is_cheating'] = True
            result['cheating_type'] = ALERT_ACTIONS[action_name]
            result['details']['reason'] = '命中异常行为类别'
        elif action_name in NORMAL_ACTIONS:
            result['details']['reason'] = '命中正常行为类别'
        else:
            result['details']['reason'] = '未配置类别，按正常处理'

        if track_id is not None:
            self.action_history[track_id].append({
                'action_name': action_name,
                'is_cheating': result['is_cheating'],
                'cheating_type': result['cheating_type'],
                'confidence': confidence
            })

            if len(self.action_history[track_id]) >= 5:
                recent = list(self.action_history[track_id])[-5:]
                same_type_count = sum(
                    1
                    for item in recent
                    if item['is_cheating'] and item['cheating_type'] == result['cheating_type']
                )

                if result['is_cheating'] and same_type_count >= 3:
                    result['confidence'] = min(1.0, confidence + 0.15)
                else:
                    result['confidence'] *= 0.5

        return result

    def update_stats(self, detection_results: List[Dict]):
        """更新统计数据。"""
        self.stats['total'] = len(detection_results)
        self.stats['cheating'] = sum(1 for r in detection_results if r['is_cheating'])
        self.stats['normal'] = self.stats['total'] - self.stats['cheating']

        for type_name in self.stats['by_type']:
            self.stats['by_type'][type_name] = 0

        for r in detection_results:
            if r['is_cheating'] and r['cheating_type'] in self.stats['by_type']:
                self.stats['by_type'][r['cheating_type']] += 1

    def get_stats(self) -> Dict:
        """获取统计数据。"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计数据。"""
        self.stats = {
            'total': 0,
            'normal': 0,
            'cheating': 0,
            'by_type': {
                '睡觉': 0,
                '使用手机': 0
            }
        }

    def get_recent_alerts(
        self,
        track_id: Optional[int] = None,
        seconds: int = 5,
    ) -> List[Dict]:
        """获取最近的异常警报。"""
        if track_id is not None:
            history = list(self.action_history.get(track_id, []))
            return [h for h in history if h['is_cheating']]
        else:
            all_alerts = []
            for tid, history in self.action_history.items():
                for h in history:
                    if h['is_cheating']:
                        h_copy = h.copy()
                        h_copy['track_id'] = tid
                        all_alerts.append(h_copy)
            return all_alerts


class SimpleTracker:
    """
    简单的人体跟踪器
    基于IOU的最近邻匹配
    """
    
    def __init__(self, max_disappear: int = 5):
        """
        初始化跟踪器
        Args:
            max_disappear: 最大消失帧数
        """
        self.max_disappear = max_disappear
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
    
    def calculate_iou(self, 
                     bbox1: List[float], 
                     bbox2: List[float]) -> float:
        """
        计算两个框的IOU
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        Returns:
            IOU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(self, detections: List[Tuple]) -> List[Tuple[int, Tuple]]:
        """
        更新跟踪
        Args:
            detections: 检测结果列表 [(bbox, conf, class_id), ...]
        Returns:
            [(track_id, detection), ...]
        """
        if len(detections) == 0:
            # 所有跟踪器消失
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappear:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []
        
        # 如果没有现有跟踪，创建新的
        if len(self.tracks) == 0:
            results = []
            for det in detections:
                self.tracks[self.next_id] = det[0]  # 保存bbox
                self.disappeared[self.next_id] = 0
                results.append((self.next_id, det))
                self.next_id += 1
            return results
        
        # 计算IOU矩阵
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid] for tid in track_ids]
        det_bboxes = [det[0] for det in detections]
        
        iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
        for i, tbox in enumerate(track_bboxes):
            for j, dbox in enumerate(det_bboxes):
                iou_matrix[i, j] = self.calculate_iou(tbox, dbox)
        
        # 匹配
        matched_tracks = set()
        matched_dets = set()
        results = []
        
        # 按IOU降序匹配
        while True:
            max_iou = 0
            max_i = -1
            max_j = -1
            
            for i in range(len(track_bboxes)):
                if i in matched_tracks:
                    continue
                for j in range(len(det_bboxes)):
                    if j in matched_dets:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i = i
                        max_j = j
            
            if max_iou < 0.3:  # IOU阈值
                break
            
            # 匹配成功
            track_id = track_ids[max_i]
            self.tracks[track_id] = det_bboxes[max_j]
            self.disappeared[track_id] = 0
            results.append((track_id, detections[max_j]))
            matched_tracks.add(max_i)
            matched_dets.add(max_j)
        
        # 未匹配的跟踪器标记为消失
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappear:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
        
        # 未匹配的检测结果创建新跟踪
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = det[0]
                self.disappeared[self.next_id] = 0
                results.append((self.next_id, det))
                self.next_id += 1
        
        return results
    
    def reset(self):
        """重置跟踪器"""
        self.next_id = 0
        self.tracks.clear()
        self.disappeared.clear()


if __name__ == '__main__':
    detector = CheatingDetector()

    result = detector.detect_cheating('using_phone', 0.92, track_id=1)
    print(f"检测结果: {result}")
