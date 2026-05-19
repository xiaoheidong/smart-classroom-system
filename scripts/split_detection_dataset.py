from __future__ import annotations

import json
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


RATIOS = {
    "train": 0.70,
    "valid": 0.15,
    "test": 0.15,
}


@dataclass
class FileSample:
    stem: str
    image_path: Path
    label_path: Path
    class_counts: Counter


@dataclass
class GroupSample:
    group_id: str
    files: List[FileSample] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def image_count(self) -> int:
        return len(self.files)

    @property
    def primary_class(self) -> int | None:
        if not self.class_counts:
            return None
        return max(
            self.class_counts,
            key=lambda cls: (self.class_counts[cls], -cls),
        )


def normalize_group_id(stem: str) -> str:
    return stem


def load_samples(dataset_root: Path) -> List[GroupSample]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    image_map = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
    label_map = {path.stem: path for path in label_dir.glob("*.txt")}

    if set(image_map) != set(label_map):
        missing_labels = sorted(set(image_map) - set(label_map))
        missing_images = sorted(set(label_map) - set(image_map))
        raise RuntimeError(
            f"image/label mismatch, missing_labels={missing_labels[:5]}, "
            f"missing_images={missing_images[:5]}"
        )

    groups: Dict[str, GroupSample] = {}
    for stem, image_path in image_map.items():
        label_path = label_map[stem]
        class_counts: Counter = Counter()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_counts[int(parts[0])] += 1

        group_id = normalize_group_id(stem)
        group = groups.setdefault(group_id, GroupSample(group_id=group_id))
        group.files.append(
            FileSample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                class_counts=class_counts,
            )
        )
        group.class_counts.update(class_counts)

    return list(groups.values())


def reset_to_train(dataset_root: Path) -> None:
    train_image_dir = dataset_root / "images" / "train"
    train_label_dir = dataset_root / "labels" / "train"

    for split in ("valid", "test"):
        for image_path in (dataset_root / "images" / split).glob("*"):
            shutil.move(str(image_path), str(train_image_dir / image_path.name))
        for label_path in (dataset_root / "labels" / split).glob("*.txt"):
            shutil.move(str(label_path), str(train_label_dir / label_path.name))


def compute_targets(total_images: int, total_class_counts: Counter) -> tuple[dict, dict]:
    target_images = {
        split: round(total_images * ratio) for split, ratio in RATIOS.items()
    }
    delta = total_images - sum(target_images.values())
    if delta != 0:
        target_images["train"] += delta

    target_classes = {
        split: {
            cls: total_class_counts[cls] * ratio for cls in total_class_counts
        }
        for split, ratio in RATIOS.items()
    }
    return target_images, target_classes


def stable_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def split_bucket(bucket: List[GroupSample]) -> dict:
    ordered = sorted(bucket, key=lambda group: stable_key(group.group_id))
    total = len(ordered)
    train_count = round(total * RATIOS["train"])
    valid_count = round(total * RATIOS["valid"])
    test_count = total - train_count - valid_count

    if total >= 3:
        if valid_count == 0:
            valid_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1

    train_end = train_count
    valid_end = train_count + valid_count
    return {
        "train": ordered[:train_end],
        "valid": ordered[train_end:valid_end],
        "test": ordered[valid_end:],
    }


def assign_groups(groups: List[GroupSample]) -> tuple[dict, dict, dict]:
    assignments = {split: [] for split in RATIOS}

    buckets: Dict[int | None, List[GroupSample]] = defaultdict(list)
    for group in groups:
        buckets[group.primary_class].append(group)

    for cls, bucket in buckets.items():
        bucket_assignment = split_bucket(bucket)
        for split in RATIOS:
            assignments[split].extend(bucket_assignment[split])

    split_images = {split: 0 for split in RATIOS}
    split_classes = {split: Counter() for split in RATIOS}
    for split in RATIOS:
        for group in assignments[split]:
            split_images[split] += group.image_count
            split_classes[split].update(group.class_counts)

    return assignments, split_images, split_classes


def ensure_directories(dataset_root: Path) -> None:
    for split in RATIOS:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def move_files(dataset_root: Path, assignments: dict) -> None:
    for split, groups in assignments.items():
        if split == "train":
            continue

        image_target_dir = dataset_root / "images" / split
        label_target_dir = dataset_root / "labels" / split
        for group in groups:
            for sample in group.files:
                shutil.move(str(sample.image_path), str(image_target_dir / sample.image_path.name))
                shutil.move(str(sample.label_path), str(label_target_dir / sample.label_path.name))


def build_summary(split_images: dict, split_classes: dict, class_names: List[str]) -> dict:
    summary = {}
    for split in RATIOS:
        summary[split] = {
            "images": split_images[split],
            "class_counts": {
                str(cls): {
                    "name": class_names[cls] if cls < len(class_names) else f"class_{cls}",
                    "objects": split_classes[split][cls],
                }
                for cls in range(len(class_names))
            },
        }
    return summary


def load_class_names(dataset_root: Path) -> List[str]:
    yaml_path = dataset_root / "data.yaml"
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    names = []
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            names.append(line[2:].strip())
    return names


def main() -> None:
    dataset_root = Path(r"d:/SY_BYSJ/smart_classroom2/data")
    ensure_directories(dataset_root)
    reset_to_train(dataset_root)
    groups = load_samples(dataset_root)
    class_names = load_class_names(dataset_root)
    assignments, split_images, split_classes = assign_groups(groups)
    move_files(dataset_root, assignments)

    summary = build_summary(split_images, split_classes, class_names)
    summary_path = dataset_root / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_saved_to={summary_path}")


if __name__ == "__main__":
    main()
