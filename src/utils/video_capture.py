"""
视频输入模块
支持：摄像头 / 视频文件 / RTSP
"""
import cv2
import numpy as np
from typing import Optional, Callable
from PyQt5.QtCore import QThread, pyqtSignal


class VideoCapture(QThread):
    """视频捕获线程"""
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, source: str = '0', frame_skip: int = 1):
        """
        初始化视频捕获
        Args:
            source: 视频源（'0'表示摄像头，或视频文件路径，或RTSP URL）
            frame_skip: 跳帧数（每N帧处理一次）
        """
        super().__init__()
        self.source = source
        self.frame_skip = frame_skip
        self.running = False
        self.cap = None
        self.frame_count = 0
        
    def open(self) -> bool:
        """打开视频源"""
        try:
            # 尝试转换为整数（摄像头索引）
            source = int(self.source)
        except ValueError:
            source = self.source
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"无法打开视频源: {self.source}")
            return False
        return True
    
    def run(self):
        """视频捕获主循环"""
        if not self.cap:
            if not self.open():
                return
                
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # 如果是视频文件，到达结尾后循环播放
                if isinstance(self.source, str) and self.source != '0':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.error_occurred.emit("视频读取失败")
                    break
            
            self.frame_count += 1
            
            # 跳帧处理
            if self.frame_count % self.frame_skip == 0:
                self.frame_ready.emit(frame)
                
        self.cap.release()
        
    def stop(self):
        """停止视频捕获"""
        self.running = False
        self.wait()
        
    def get_fps(self) -> float:
        """获取视频帧率"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0
        
    def get_resolution(self) -> tuple:
        """获取视频分辨率"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (640, 480)
        
    def set_resolution(self, width: int, height: int):
        """设置摄像头分辨率"""
        if self.cap and isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


class VideoProcessor:
    """视频处理器（非线程版，用于单帧处理）"""
    def __init__(self, source: str = '0'):
        self.source = source
        self.cap = None
        
    def open(self) -> bool:
        """打开视频源"""
        try:
            source = int(self.source)
        except ValueError:
            source = self.source
            
        self.cap = cv2.VideoCapture(source)
        return self.cap.isOpened()
    
    def read(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
            self.cap = None
