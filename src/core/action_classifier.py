"""
行为分类模块
基于人物裁剪图像的 CNN 行为分类器
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTION_CLASSIFIER_MODEL,
    ACTION_CLASSES,
    ACTION_DISPLAY_NAMES,
    DEVICE,
    TRAINED_DIR,
    resolve_action_classifier_path,
)


class ActionClassifier:
    """基于人物裁剪图像的行为分类器。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = DEVICE,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = resolve_action_classifier_path() or os.path.join(
                TRAINED_DIR, ACTION_CLASSIFIER_MODEL
            )

        self.model_path = model_path
        self.class_names = [ACTION_CLASSES[i] for i in sorted(ACTION_CLASSES)]
        self.image_size = 224
        self.model = None
        self.transform = None
        self._stub = False

        self._load_model()

    def _create_model(self, num_classes: int) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self._stub = True
            print(
                f"警告: 未找到行为分类模型: {self.model_path}\n"
                "  作弊检测将使用占位分类（界面可预览，行为结果无实际意义）。\n"
                "  请将 action_classifier_cnn_best.pth 放在 models/trained/ 或 trained 下某次 run 子目录中。"
            )
            return

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            self.class_names = checkpoint.get("class_names", self.class_names)
            self.image_size = checkpoint.get("image_size", self.image_size)

        self.model = self._create_model(len(self.class_names))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"加载行为分类模型: {self.model_path}")
        print(f"行为类别: {self.class_names}")

    def get_display_name(self, action_name: str) -> str:
        """将模型类别名转换为界面展示名。"""
        return ACTION_DISPLAY_NAMES.get(action_name, action_name)

    def preprocess_crop(self, person_crop: np.ndarray) -> torch.Tensor:
        """预处理人物裁剪图。"""
        if person_crop is None or person_crop.size == 0:
            raise ValueError("输入的人物裁剪图为空")

        rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor

    def classify(self, person_crop: np.ndarray) -> Tuple[int, float, str]:
        """分类单个人物裁剪图。"""
        if self._stub:
            idx = 1  # raise_head，仅占位便于界面跑通
            return idx, 0.5, self.class_names[idx]

        inputs = self.preprocess_crop(person_crop)

        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        class_id = predicted.item()
        confidence_val = confidence.item()
        action_name = self.class_names[class_id]
        return class_id, confidence_val, action_name

    def classify_with_display(self, person_crop: np.ndarray) -> Dict[str, object]:
        """分类并返回原始标签与展示标签。"""
        class_id, confidence, action_name = self.classify(person_crop)
        return {
            "class_id": class_id,
            "confidence": confidence,
            "action_name": action_name,
            "display_name": self.get_display_name(action_name),
        }


if __name__ == "__main__":
    classifier = ActionClassifier()
    test_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = classifier.classify_with_display(test_crop)
    print(
        f"分类结果: {result['display_name']} "
        f"(ID: {result['class_id']}), 置信度: {result['confidence']:.2f}"
    )
