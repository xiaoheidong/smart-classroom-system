"""
可视化工具模块
用于绘制检测框、关键点、统计信息等
"""
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    """可视化类"""
    
    def __init__(self):
        self.colors = {
            'normal': (0, 255, 0),      # 绿色
            'cheating': (0, 0, 255),    # 红色
            'warning': (0, 255, 255),   # 黄色
            'bbox': (255, 0, 0),        # 蓝色
        }
        self.font_path = self._get_font_path()
    
    def _get_font_path(self) -> Optional[str]:
        """获取系统中的中文字体路径。"""
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        background_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 6,
    ) -> np.ndarray:
        """使用 PIL 在图像上绘制支持中文的文字。"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(image)

        if self.font_path:
            font = ImageFont.truetype(self.font_path, font_size)
        else:
            font = ImageFont.load_default()

        x, y = position
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)

        if background_color is not None:
            bg_color = (
                int(background_color[2]),
                int(background_color[1]),
                int(background_color[0]),
            )
            draw.rectangle(
                (
                    left - padding,
                    top - padding,
                    right + padding,
                    bottom + padding,
                ),
                fill=bg_color,
            )

        text_color = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y), text, font=font, fill=text_color)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _measure_text(self, text: str, font_size: int = 22) -> Tuple[int, int]:
        """用与 _draw_text 相同字体测量文本宽高（用于中文标签定位）。"""
        rgb = np.zeros((80, 2048, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        if self.font_path:
            font = ImageFont.truetype(self.font_path, font_size)
        else:
            font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)

    def draw_bbox(self, 
                  frame: np.ndarray, 
                  bbox: List[float], 
                  label: str,
                  confidence: float,
                  color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        绘制检测框和标签
        Args:
            frame: 输入图像
            bbox: [x1, y1, x2, y2] 检测框坐标
            label: 标签文本
            confidence: 置信度
            color: 框的颜色
        Returns:
            绘制后的图像
        """
        if color is None:
            color = self.colors['bbox']
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label_text = f"{label}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - text_h - 10), 
            (x1 + text_w, y1), 
            color, 
            -1
        )
        
        # 中文标签使用 PIL 绘制，避免 OpenCV 字体显示成问号。
        frame = self._draw_text(
            frame,
            label_text,
            (x1 + 4, y1 - text_h - 8),
            font_size=22,
            color=(255, 255, 255),
            background_color=color,
            padding=4,
        )
        
        return frame
    
    def draw_face_box(self,
                      frame: np.ndarray,
                      face_rect: Tuple[int, int, int, int],
                      name: Optional[str] = None,
                      confidence: Optional[float] = None,
                      is_live: Optional[bool] = None) -> np.ndarray:
        """
        绘制人脸框
        Args:
            frame: 输入图像
            face_rect: (left, top, right, bottom)
            name: 人名
            confidence: 识别置信度
            is_live: 是否为活体
        Returns:
            绘制后的图像
        """
        left, top, right, bottom = face_rect
        
        # 根据活体检测结果选择颜色
        if is_live is None:
            color = (255, 165, 0)  # 橙色 - 未检测
        elif is_live:
            color = self.colors['normal']  # 绿色 - 活体
        else:
            color = self.colors['cheating']  # 红色 - 非活体
        
        # 绘制框
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # 绘制标签（使用 PIL，与 draw_bbox 一致，支持中文姓名）
        if name:
            label = name
            if confidence is not None:
                label += f" ({confidence:.2f})"
            if is_live is not None:
                label += " [活体]" if is_live else " [非活体]"

            fs = 20
            text_w, text_h = self._measure_text(label, fs)
            label_y = max(0, top - text_h - 10)
            frame = self._draw_text(
                frame,
                label,
                (left + 4, label_y),
                font_size=fs,
                color=(255, 255, 255),
                background_color=color,
                padding=4,
            )

        return frame
    
    def draw_stats(self,
                   frame: np.ndarray,
                   stats: Dict[str, int],
                   position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        绘制统计信息
        Args:
            frame: 输入图像
            stats: 统计字典
            position: 起始位置
        Returns:
            绘制后的图像
        """
        x, y = position
        for key, value in stats.items():
            text = f"{key}: {value}"
            frame = self._draw_text(
                frame,
                text,
                (x, y),
                font_size=24,
                color=(255, 255, 255),
                background_color=(0, 0, 0),
                padding=4,
            )
            y += 30
        
        return frame
    
    def draw_attendance_list(self,
                             frame: np.ndarray,
                             present_students: List[str],
                             absent_students: List[str],
                             width: int = 300) -> np.ndarray:
        """
        绘制考勤名单
        Args:
            frame: 原图像
            present_students: 已签到学生列表
            absent_students: 未到学生列表
            width: 侧边栏宽度
        Returns:
            合成后的图像
        """
        h, w = frame.shape[:2]
        
        # 创建侧边栏
        sidebar = np.zeros((h, width, 3), dtype=np.uint8)
        sidebar[:] = (50, 50, 50)
        
        y_offset = 30
        
        # 绘制标题
        cv2.putText(
            sidebar,
            "考勤名单",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        y_offset += 40
        
        # 绘制已签到学生
        cv2.putText(
            sidebar,
            f"已签到 ({len(present_students)})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors['normal'],
            2
        )
        y_offset += 25
        
        for student in present_students[-10:]:  # 只显示最近10个
            cv2.putText(
                sidebar,
                f"  {student}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            y_offset += 20
        
        y_offset += 10
        
        # 绘制未到学生
        cv2.putText(
            sidebar,
            f"未到 ({len(absent_students)})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors['cheating'],
            2
        )
        y_offset += 25
        
        for student in absent_students[:10]:  # 只显示前10个
            cv2.putText(
                sidebar,
                f"  {student}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1
            )
            y_offset += 20
        
        # 合并图像
        result = np.hstack([frame, sidebar])
        return result


def resize_frame(frame: np.ndarray, 
                 max_width: int = 1280, 
                 max_height: int = 720) -> np.ndarray:
    """
    调整帧大小（保持宽高比）
    Args:
        frame: 输入图像
        max_width: 最大宽度
        max_height: 最大高度
    Returns:
        调整后的图像
    """
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # 只缩小，不放大
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    return frame
