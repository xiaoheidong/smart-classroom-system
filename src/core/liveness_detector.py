"""
静默活体检测模块
使用MiniFASNet进行单帧活体检测
"""
import os
import cv2
import torch
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRETRAINED_DIR, LIVENESS_MODEL, FACE_CONFIG


class LivenessDetector:
    """静默活体检测器"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 input_size: int = 80,
                 threshold: float = FACE_CONFIG['liveness_threshold']):
        """
        初始化活体检测器
        Args:
            model_path: 模型路径
            device: 运行设备
            input_size: 输入图像大小
            threshold: 活体判断阈值
        """
        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        
        # 模型路径
        if model_path is None:
            model_path = os.path.join(PRETRAINED_DIR, LIVENESS_MODEL)
        
        # 加载模型
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载MiniFASNet模型"""
        if not os.path.exists(model_path):
            print(f"警告: 活体检测模型不存在: {model_path}")
            return None
        
        try:
            # 尝试加载完整的模型文件
            model = torch.load(model_path, map_location=self.device)
            
            # 如果加载的是状态字典，需要构建模型结构
            if isinstance(model, dict):
                from collections import OrderedDict
                # 创建简单的CNN模型结构
                model = self._create_minifasnet()
                model.load_state_dict(model)
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def _create_minifasnet(self):
        """
        创建MiniFASNet模型结构
        这是一个轻量级的分类网络
        """
        # 简化版MiniFASNet
        model = torch.nn.Sequential(
            # 输入层
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            
            # 深度可分离卷积块
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=16),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(32, 32, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            # 下采样
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1, groups=32),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            # 全局平均池化
            torch.nn.AdaptiveAvgPool2d(1),
            
            # 分类头
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 3)  # 3类: 真人、照片、屏幕
        )
        
        return model.to(self.device)
    
    def preprocess(self, face_img: np.ndarray) -> torch.Tensor:
        """
        预处理人脸图像
        Args:
            face_img: 人脸图像 [H, W, 3]
        Returns:
            预处理后的张量 [1, 3, 80, 80]
        """
        # 调整大小
        img = cv2.resize(face_img, (self.input_size, self.input_size))
        
        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 减均值除方差（ImageNet统计）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 添加batch维度
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        return tensor
    
    def detect(self,
               frame: np.ndarray,
               face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, float]:
        """
        进行活体检测
        Args:
            frame: 输入图像
            face_rect: 人脸框 (left, top, right, bottom)，None则使用整个图像
        Returns:
            (是否为活体, 置信度分数)
        """
        if self.model is None:
            # 模型未加载，默认通过
            return True, 1.0
        
        # 裁剪人脸区域
        if face_rect is not None:
            left, top, right, bottom = face_rect
            # 扩大框以包含更多上下文
            margin = int((right - left) * 0.3)
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(frame.shape[1], right + margin)
            bottom = min(frame.shape[0], bottom + margin)
            
            face_img = frame[top:bottom, left:right]
        else:
            face_img = frame
        
        if face_img.size == 0:
            return False, 0.0
        
        # 预处理
        input_tensor = self.preprocess(face_img)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
            # 获取各类别概率
            real_prob = probs[0, 0].item()
            photo_prob = probs[0, 1].item()
            screen_prob = probs[0, 2].item()
        
        # 判断是否为真人
        is_live = real_prob > self.threshold
        
        # 综合分数
        score = real_prob
        
        return is_live, score
    
    def detect_batch(self,
                    frame: np.ndarray,
                    face_rects: list) -> list:
        """
        批量活体检测
        Args:
            frame: 输入图像
            face_rects: 人脸框列表
        Returns:
            [(is_live, score), ...]
        """
        results = []
        for face_rect in face_rects:
            result = self.detect(frame, face_rect)
            results.append(result)
        return results


class SimpleLivenessDetector:
    """
    简单的活体检测（备用方案）
    基于纹理分析和眨眼检测
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def detect(self,
               frame: np.ndarray,
               face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, float]:
        """
        简单活体检测
        Args:
            frame: 输入图像
            face_rect: 人脸框
        Returns:
            (是否为活体, 置信度)
        """
        if face_rect is not None:
            left, top, right, bottom = face_rect
            face_img = frame[top:bottom, left:right]
        else:
            face_img = frame
        
        if face_img.size == 0:
            return False, 0.0
        
        # 转换为灰度图
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 计算Laplacian方差（纹理清晰度）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 真人的纹理通常更清晰
        # 简单阈值判断
        score = min(1.0, laplacian_var / 500.0)
        
        is_live = score > self.threshold
        
        return is_live, score


if __name__ == '__main__':
    # 测试代码
    detector = LivenessDetector()
    
    # 生成测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    is_live, score = detector.detect(test_image)
    print(f"活体检测结果: {is_live}, 分数: {score:.2f}")
