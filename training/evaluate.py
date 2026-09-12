"""
RT-DETR Evaluation & Benchmark Metrics Script
---------------------------------------------
Evaluates the trained model weights on the unseen test set.
Computes:
- mAP@50 (Mean Average Precision at IoU 0.50)
- mAP@50-95 (Mean Average Precision over IoU range [0.50, 0.95])
- Per-class Precision, Recall, and F1-scores
- Outputs metrics formatted for inclusion in the 2-Page Engineering Memo.
"""

import sys
from pathlib import Path
from ultralytics import RTDETR

def run_evaluation(weights_path: str = None):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_yaml = project_root / "dataset" / "data.yaml"

    if weights_path is None:
        weights_path = project_root / "weights" / "best.pt"
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        print(f"[!] Target weights not found at: {weights_path}")
        print("[!] Defaulting to pretrained backbone 'rtdetr-l.pt' for baseline benchmarking...")
        weights_path = "rtdetr-l.pt"

    print("=" * 65)
    print("RAP EVALUATION PIPELINE: RT-DETR BENCHMARK ON TEST SPLIT")
    print(f"Weights: {weights_path}")
    print(f"Dataset: {data_yaml}")
    print("=" * 65)

    model = RTDETR(str(weights_path))

    # Evaluate on the holdout test set
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.60,
        device="0" if model.device.type == "cuda" else "cpu",
        project=str(project_root / "runs"),
        name="rtdetr_evaluation",
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY (For 2-Page Engineering Memo)")
    print("=" * 65)
    print(f"Overall mAP@50:       {metrics.box.map50 * 100:.2f}%")
    print(f"Overall mAP@50-95:    {metrics.box.map * 100:.2f}%")
    print(f"Overall Mean Precision:{metrics.box.mp * 100:.2f}%")
    print(f"Overall Mean Recall:   {metrics.box.mr * 100:.2f}%")
    print("-" * 65)
    print("Per-Class Performance Breakdown:")
    class_names = ["hard-hat", "safety-vest", "person"]
    for i, name in enumerate(class_names):
        if i < len(metrics.box.maps):
            print(f" • Class [{name:12}]: mAP@50 = {metrics.box.all_ap[i][0] * 100:.2f}% | mAP@50-95 = {metrics.box.maps[i] * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    weights_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(weights_arg)
