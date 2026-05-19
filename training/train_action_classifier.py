"""
行为分类模型训练脚本

适配当前 `data/` 目录中的 YOLO 检测标注数据：
1. 读取 `data.yaml` 中的类别定义
2. 根据 `labels/train|valid|test/*.txt` 里的框标注裁出人物区域
3. 使用图像分类模型训练行为分类器

示例:
    python training/train_action_classifier.py --data_dir ./data --epochs 30
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEVICE, TRAINED_DIR


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int) -> None:
    """固定随机种子，便于复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_data_yaml(yaml_path: Path) -> List[str]:
    """从简单的 data.yaml 中解析类别名。"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"未找到类别配置文件: {yaml_path}")

    class_names: List[str] = []
    in_names_block = False

    with yaml_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("names:"):
                in_names_block = True
                continue

            if in_names_block:
                if line.startswith("- "):
                    class_names.append(line[2:].strip())
                    continue

                if ":" in line:
                    break

    if not class_names:
        raise ValueError(f"无法从 {yaml_path} 中解析类别名")

    return class_names


def resolve_labels_dir(data_dir: Path, split: str) -> Path:
    """定位标注目录。"""
    candidates = [
        data_dir / "labels" / split,
        data_dir / split / "labels",
        data_dir / split,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            txt_files = list(candidate.glob("*.txt"))
            if txt_files:
                return candidate
    raise FileNotFoundError(f"未找到 {split} 集标注目录，请检查 {data_dir}")


def resolve_image_path(data_dir: Path, split: str, stem: str) -> Optional[Path]:
    """根据标注文件名查找对应图片。"""
    candidate_roots = [
        data_dir / "images" / split,
        data_dir / split / "images",
        data_dir / split,
        data_dir / "VOC" / split,
        data_dir,
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:
    """将 YOLO 归一化框转换为像素框。"""
    box_w = width * image_width
    box_h = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio

    x1 = max(0, int(center_x - box_w / 2 - pad_w))
    y1 = max(0, int(center_y - box_h / 2 - pad_h))
    x2 = min(image_width, int(center_x + box_w / 2 + pad_w))
    y2 = min(image_height, int(center_y + box_h / 2 + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


class DetectionCropDataset(Dataset):
    """将检测框动态裁成分类样本的数据集。"""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        class_names: Sequence[str],
        transform: transforms.Compose,
    ):
        self.data_dir = data_dir
        self.split = split
        self.class_names = list(class_names)
        self.transform = transform
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f"{split} 集没有可用样本，请检查图片和标注是否完整")

    def _build_samples(self) -> List[Tuple[str, int, Tuple[int, int, int, int]]]:
        labels_dir = resolve_labels_dir(self.data_dir, self.split)
        label_files = sorted(labels_dir.glob("*.txt"))
        samples: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        missing_images = 0

        print(f"扫描 {self.split} 集标注: {labels_dir}")

        for label_file in tqdm(label_files, desc=f"构建{self.split}样本"):
            image_path = resolve_image_path(self.data_dir, self.split, label_file.stem)
            if image_path is None:
                missing_images += 1
                continue

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(self.class_names):
                        continue

                    x_center, y_center, width, height = map(float, parts[1:])
                    bbox = yolo_to_xyxy(
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if bbox is None:
                        continue

                    samples.append((str(image_path), class_id, bbox))

        print(f"{self.split} 集样本数: {len(samples)}")
        if missing_images > 0:
            print(f"  警告: 有 {missing_images} 个标注文件未找到对应图片")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, class_id, bbox = self.samples[idx]
        x1, y1, x2, y2 = bbox

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            crop = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, class_id


def create_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """创建训练与验证数据增强。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def create_model(num_classes: int, pretrained: bool) -> nn.Module:
    """创建图像分类模型。"""
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def calculate_class_weights(dataset: DetectionCropDataset, num_classes: int) -> torch.Tensor:
    """根据训练集分布计算类别权重，缓解类别不平衡。"""
    counts = Counter(class_id for _, class_id, _ in dataset.samples)
    total = sum(counts.values())
    weights = []

    for class_id in range(num_classes):
        count = counts.get(class_id, 1)
        weights.append(total / (num_classes * count))

    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="训练")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total, 1):.2f}%",
        })

    return total_loss / max(len(dataloader), 1), 100.0 * correct / max(total, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """在验证/测试集上评估。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return (
        total_loss / max(len(dataloader), 1),
        100.0 * correct / max(total, 1),
        all_preds,
        all_labels,
    )


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    class_names: Sequence[str],
    image_size: int,
    best_acc: float,
) -> None:
    """保存模型与元数据。"""
    payload = {
        "model_name": "resnet18",
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "image_size": image_size,
        "best_val_acc": best_acc,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, save_path)


def save_training_plots(history: Dict[str, List[float]], output_dir: Path) -> None:
    """保存更适合论文展示的训练过程曲线图。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        return

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val_acc = history["val_acc"][best_epoch - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#f7f8fa")
    ax1, ax2 = axes

    ax1.plot(epochs, history["train_loss"], marker="o", linewidth=2.2, color="#2f6fed", label="训练损失")
    ax1.plot(epochs, history["val_loss"], marker="o", linewidth=2.2, color="#f08c2e", label="验证损失")
    ax1.set_xlabel("训练轮次 Epoch")
    ax1.set_ylabel("损失值 Loss")
    ax1.set_title("训练集与验证集损失变化曲线", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=True)

    ax2.plot(epochs, history["train_acc"], marker="o", linewidth=2.2, color="#1f9d55", label="训练准确率")
    ax2.plot(epochs, history["val_acc"], marker="o", linewidth=2.2, color="#c0392b", label="验证准确率")
    ax2.scatter(best_epoch, best_val_acc, s=90, color="#c0392b", zorder=5)
    ax2.annotate(
        f"最佳验证准确率\nEpoch {best_epoch}: {best_val_acc:.2f}%",
        xy=(best_epoch, best_val_acc),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e8", ec="#f08c2e", alpha=0.95),
    )
    ax2.set_xlabel("训练轮次 Epoch")
    ax2.set_ylabel("准确率 Accuracy (%)")
    ax2.set_title("训练集与验证集准确率变化曲线", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=True)

    fig.suptitle("行为分类模型训练过程可视化", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_confusion_matrix_plot(
    matrix,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """保存论文风格的混淆矩阵图。"""
    fig = plt.figure(figsize=(10, 8), facecolor="#f7f8fa")
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("测试集混淆矩阵", fontsize=15, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2 if matrix.size > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.ylabel("真实类别", fontsize=11)
    plt.xlabel("预测类别", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_class_accuracy_plot(
    matrix,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """保存各类别识别率柱状图，更适合论文展示。"""
    class_totals = matrix.sum(axis=1)
    class_correct = np.diag(matrix)
    class_acc = np.divide(
        class_correct,
        np.maximum(class_totals, 1),
        out=np.zeros_like(class_correct, dtype=float),
        where=np.maximum(class_totals, 1) != 0,
    ) * 100

    fig = plt.figure(figsize=(11, 6), facecolor="#f7f8fa")
    ax = plt.gca()
    bars = ax.bar(class_names, class_acc, color="#4c72b0", edgecolor="#2c3e50", alpha=0.9)

    ax.set_ylim(0, 100)
    ax.set_ylabel("类别识别率 (%)")
    ax.set_xlabel("行为类别")
    ax.set_title("各类别识别率对比图", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, acc in zip(bars, class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def train(args: argparse.Namespace) -> None:
    """主训练函数。"""
    print("=" * 60)
    print("行为分类模型训练（基于检测框裁剪）")
    print("=" * 60)

    set_seed(args.seed)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(args.yaml_path).resolve() if args.yaml_path else data_dir / "data.yaml"
    class_names = parse_data_yaml(yaml_path)

    print(f"数据目录: {data_dir}")
    print(f"类别数: {len(class_names)}")
    print(f"类别列表: {class_names}")

    train_transform, eval_transform = create_transforms(args.image_size)
    train_dataset = DetectionCropDataset(data_dir, "train", class_names, train_transform)
    val_dataset = DetectionCropDataset(data_dir, "valid", class_names, eval_transform)

    test_dataset = None
    try:
        test_dataset = DetectionCropDataset(data_dir, "test", class_names, eval_transform)
    except Exception as exc:
        print(f"未加载 test 集，最终评估将使用 valid 集: {exc}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = create_model(num_classes=len(class_names), pretrained=args.pretrained)
    model.to(device)

    print("\n模型结构:")
    print(model)
    print(f"总参数量: {sum(p.numel() for p in model.parameters())}")

    class_weights = calculate_class_weights(train_dataset, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_path = output_dir / "action_classifier_cnn_best.pth"
    final_path = output_dir / "action_classifier_cnn.pth"
    metadata_path = output_dir / "action_classifier_cnn_meta.json"

    print(f"\n开始训练 {args.epochs} epochs...")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, best_path, class_names, args.image_size, best_acc)
            print(f"  [BEST] 保存最佳模型: {best_path.name} (准确率: {best_acc:.2f}%)")

    save_checkpoint(model, final_path, class_names, args.image_size, best_acc)

    eval_loader = test_loader if test_loader is not None else val_loader
    eval_name = "test" if test_loader is not None else "valid"
    eval_loss, eval_acc, preds, labels = evaluate(model, eval_loader, criterion, device)

    print("\n" + "=" * 60)
    print(f"最终评估（{eval_name} 集）")
    print("=" * 60)
    print(f"Loss={eval_loss:.4f}, Acc={eval_acc:.2f}%")
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    confusion = confusion_matrix(labels, preds)
    print("混淆矩阵:")
    print(confusion)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "resnet18",
                "class_names": class_names,
                "num_classes": len(class_names),
                "image_size": args.image_size,
                "best_val_acc": best_acc,
                "final_eval_split": eval_name,
                "final_eval_acc": eval_acc,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    save_training_plots(history, output_dir)
    save_confusion_matrix_plot(
        confusion,
        class_names,
        output_dir / "confusion_matrix.png",
    )
    save_class_accuracy_plot(
        confusion,
        class_names,
        output_dir / "class_accuracy.png",
    )

    print("\n训练完成!")
    print(f"最佳模型: {best_path}")
    print(f"最终模型: {final_path}")
    print(f"元数据: {metadata_path}")
    print(f"训练曲线图: {output_dir / 'training_curves.png'}")
    print(f"混淆矩阵图: {output_dir / 'confusion_matrix.png'}")
    print(f"类别识别率图: {output_dir / 'class_accuracy.png'}")
    print(f"最佳验证准确率: {best_acc:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练行为分类模型（YOLO 标注裁剪版）")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--yaml_path", type=str, default="", help="类别配置文件路径，默认 data_dir/data.yaml")
    parser.add_argument("--output_dir", type=str, default=TRAINED_DIR, help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--image_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 工作线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="是否使用 ImageNet 预训练权重（首次使用可能会下载）",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
