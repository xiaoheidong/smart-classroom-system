"""
模型导出脚本
将训练好的模型导出为TorchScript格式，便于部署

使用方法:
    python scripts/export_models.py

导出模型:
    - YOLOv11n 人体检测模型
    - CNN 行为分类模型
"""
import os
import sys
import torch
import argparse
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    PRETRAINED_DIR, TRAINED_DIR,
    YOLOV11N_MODEL,
    ACTION_CLASSIFIER_MODEL
)
from ultralytics import YOLO


def export_yolo_model(model_path, output_path, imgsz=640):
    """
    导出YOLO模型为TorchScript
    Args:
        model_path: 模型路径
        output_path: 输出路径
        imgsz: 输入图像大小
    """
    print(f"导出YOLO模型: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # 导出为TorchScript
        model.export(format='torchscript', imgsz=imgsz)
        
        # 移动并重命名
        exported_path = model_path.replace('.pt', '.torchscript.pt')
        if os.path.exists(exported_path):
            os.rename(exported_path, output_path)
        
        print(f"  ✓ 导出成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        return False


def export_action_classifier(model_path, output_path):
    """
    导出行为分类模型
    Args:
        model_path: 模型路径
        output_path: 输出路径
    """
    print(f"导出行为分类模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError('分类模型格式不正确，缺少 state_dict')

        class_names = checkpoint.get('class_names', [])
        if not class_names:
            raise ValueError('分类模型缺少 class_names 元数据')

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # 示例输入
        image_size = checkpoint.get('image_size', 224)
        example_input = torch.randn(1, 3, image_size, image_size)
        
        # 导出为TorchScript
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
        
        print(f"  ✓ 导出成功: {output_path}")
        
        # 验证导出的模型
        loaded = torch.jit.load(output_path)
        test_output = loaded(example_input)
        print(f"  验证输出形状: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='导出模型为TorchScript')
    parser.add_argument('--output_dir', type=str, default='./models/trained',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型导出工具")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 导出YOLOv11n（人体检测）
    yolo_model = os.path.join(PRETRAINED_DIR, YOLOV11N_MODEL)
    if os.path.exists(yolo_model):
        output = os.path.join(args.output_dir, 'yolov11n_person.torchscript.pt')
        export_yolo_model(yolo_model, output)
    else:
        print(f"跳过YOLOv11n导出: 模型不存在 ({yolo_model})")
    
    print()
    
    # 导出行为分类器
    action_model = os.path.join(TRAINED_DIR, ACTION_CLASSIFIER_MODEL)
    if os.path.exists(action_model):
        output = os.path.join(args.output_dir, 'action_classifier_cnn.torchscript.pt')
        export_action_classifier(action_model, output)
    else:
        print(f"跳过行为分类器导出: 模型不存在 ({action_model})")
        print("提示: 请先运行 training/train_action_classifier.py 训练模型")
    
    print()
    print("=" * 60)
    print("导出完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
