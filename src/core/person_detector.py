"""
人体检测模块
使用YOLOv11n进行轻量化人体检测
"""
import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PRETRAINED_DIR, TRAINED_DIR, 
    YOLOV11N_MODEL, YOLOV11N_PERSON_MODEL,
    DEVICE, BATCH_SIZE, IMAGE_SIZE
)


class PersonDetector:
    """人体检测器"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = DEVICE):
        """
        初始化人体检测器
        Args:
            model_path: 模型路径，默认使用YOLOv11n预训练模型
            confidence_threshold: 置信度阈值
            device: 运行设备
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        if not torch.cuda.is_available() and "cuda" in str(self.device).lower():
            self.device = "cpu"
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(PRETRAINED_DIR, YOLOV11N_MODEL)
            # 如果没有本地模型，使用ultralytics自动下载
            if not os.path.exists(model_path):
                model_path = 'yolo11n.pt'
        
        self.model = YOLO(model_path)
        # 必须用 self.device（已回退 CPU）；勿使用参数 device，否则 CPU 版 torch 会因 cuda:0 报错
        self.model.to(self.device)
        
    def detect(self, 
               frame: np.ndarray,
               classes: Optional[List[int]] = None) -> List[Tuple[List[float], float, int]]:
        """
        检测人体
        Args:
            frame: 输入图像 (BGR格式)
            classes: 指定检测类别，默认只检测person类（COCO中id=0）
        Returns:
            检测结果列表 [(bbox, confidence, class_id), ...]
            bbox: [x1, y1, x2, y2]
        """
        if classes is None:
            classes = [0]  # COCO中person类的id是0
        
        # 运行检测
        results = self.model(
            frame,
            classes=classes,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append((box.tolist(), float(conf), int(cls_id)))
        
        return detections
    
    def detect_batch(self,
                     frames: List[np.ndarray],
                     classes: Optional[List[int]] = None) -> List[List[Tuple]]:
        """
        批量检测（用于处理视频序列）
        Args:
            frames: 图像列表
            classes: 指定检测类别
        Returns:
            每帧的检测结果列表
        """
        if classes is None:
            classes = [0]
        
        results = self.model(
            frames,
            classes=classes,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            batch=len(frames)
        )
        
        batch_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    detections.append((box.tolist(), float(conf), int(cls_id)))
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def export_torchscript(self, output_path: str):
        """
        导出TorchScript模型
        Args:
            output_path: 输出路径
        """
        self.model.export(format='torchscript', imgsz=IMAGE_SIZE)
        print(f"模型已导出到: {output_path}")


if __name__ == '__main__':
    # 测试代码
    import cv2
    
    detector = PersonDetector()
    
    # 测试图片
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_image)
    
    print(f"检测到 {len(detections)} 个人")
    for bbox, conf, cls_id in detections:
        print(f"  bbox: {bbox}, conf: {conf:.2f}")
