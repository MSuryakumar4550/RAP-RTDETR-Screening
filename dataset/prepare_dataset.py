"""
Dataset Preparation and Verification Script
-------------------------------------------
Domain: Construction Site Safety / Worker PPE
Classes:
    0: hard-hat (Non-COCO)
    1: safety-vest (Non-COCO)
    2: person (Reference class)

This script:
1. Downloads or ingests the Construction PPE dataset in YOLO/RT-DETR format.
2. Performs data integrity checks (matching image-label pairs, valid normalized coordinates).
3. Computes class distribution and split statistics for the 2-Page Technical Memo.
"""

import os
import glob
from pathlib import Path
from collections import Counter

CLASS_NAMES = {0: "hard-hat", 1: "safety-vest", 2: "person"}

def audit_split(split_name: str, base_dir: Path):
    """
    Audits a split directory (train, valid, or test) to verify image-label pairs
    and extract class-wise frequency metrics.
    """
    images_dir = base_dir / split_name / "images"
    labels_dir = base_dir / split_name / "labels"

    if not images_dir.exists():
        return {
            "split": split_name,
            "images_count": 0,
            "labels_count": 0,
            "missing_labels": 0,
            "corrupt_boxes": 0,
            "class_counts": Counter(),
        }

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))

    class_counts = Counter()
    missing_labels = 0
    corrupt_boxes = 0

    for img_path in image_files:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    # Validation: normalized values must be strictly in [0.0, 1.0]
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        corrupt_boxes += 1
                    else:
                        class_counts[cls_id] += 1
                except ValueError:
                    corrupt_boxes += 1

    return {
        "split": split_name,
        "images_count": len(image_files),
        "labels_count": len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0,
        "missing_labels": missing_labels,
        "corrupt_boxes": corrupt_boxes,
        "class_counts": class_counts,
    }


def generate_dataset_report(dataset_dir: str = "."):
    """
    Scans the entire dataset directory and outputs a comprehensive
    distribution report formatted for the Technical Memo.
    """
    base_dir = Path(dataset_dir)
    splits = ["train", "valid", "test"]

    print("=" * 65)
    print("CONSTRUCTION PPE DATASET - SPLIT & DISTRIBUTION AUDIT")
    print("=" * 65)

    total_images = 0
    total_annotations = Counter()

    for split in splits:
        stats = audit_split(split, base_dir)
        total_images += stats["images_count"]
        total_annotations.update(stats["class_counts"])

        print(f"\n📂 Split: {split.upper()}")
        print(f"   • Total Images:       {stats['images_count']}")
        print(f"   • Missing Labels:     {stats['missing_labels']}")
        print(f"   • Corrupted Boxes:    {stats['corrupt_boxes']}")
        print("   • Class Frequencies:")
        for cls_id, count in sorted(stats["class_counts"].items()):
            name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
            print(f"       - {name} (Class {cls_id}): {count}")

    print("\n" + "=" * 65)
    print(f"SUMMARY: Total Images: {total_images}")
    print("Aggregate Class Distribution:")
    for cls_id, count in sorted(total_annotations.items()):
        name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        print(f"   • {name}: {count} annotations")
    print("=" * 65)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_dataset_report(current_dir)
