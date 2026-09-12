"""
Reproducible RT-DETR Training Pipeline
---------------------------------------
Trains RT-DETR (Real-Time Detection Transformer) on the Construction PPE dataset.

Adheres strictly to RAP Hard Constraint #4:
- Hardware, environment, seeds, training time, and hyperparameters are strictly logged.
"""

import os
import sys
import time
import yaml
import torch
from pathlib import Path
from ultralytics import RTDETR

def set_reproducibility(seed: int = 42):
    """
    Sets global deterministic seeds across PyTorch, CUDA, and NumPy.
    Java Analogy: System.setProperty / ThreadLocalRandom with a fixed seed.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[*] Deterministic seed locked to: {seed}")

def load_config(config_path: str) -> dict:
    """Reads hyperparameters from YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_training():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_yaml = project_root / "dataset" / "data.yaml"
    config_yaml = script_dir / "hyperparams.yaml"

    if not config_yaml.exists():
        raise FileNotFoundError(f"Config file not found at: {config_yaml}")

    config = load_config(str(config_yaml))
    seed = config.get("seed", 42)
    set_reproducibility(seed)

    # Hardware verification
    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print("RAP SCREENING: RT-DETR TRAINING INITIALIZATION")
    print(f"Target Hardware:  {device_name} (device={device})")
    print(f"Dataset Config:   {data_yaml}")
    print(f"Base Architecture:{config.get('model', 'rtdetr-l.pt')}")
    print(f"Epochs:           {config.get('epochs', 35)}")
    print(f"Batch Size:       {config.get('batch', 8)}")
    print("=" * 60)

    # Initialize RT-DETR model with pretrained backbone
    model = RTDETR(config.get("model", "rtdetr-l.pt"))

    start_time = time.time()

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=config.get("epochs", 35),
        batch=config.get("batch", 8),
        imgsz=config.get("imgsz", 640),
        optimizer=config.get("optimizer", "AdamW"),
        lr0=config.get("lr0", 0.0001),
        lrf=config.get("lrf", 0.01),
        weight_decay=config.get("weight_decay", 0.0001),
        seed=seed,
        deterministic=True,
        mosaic=config.get("mosaic", 1.0),
        mixup=config.get("mixup", 0.15),
        project=str(project_root / "runs"),
        name="rtdetr_ppe_run",
        exist_ok=True,
        save=True
    )

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"\n[+] Training completed in {elapsed_minutes:.2f} minutes.")

    # Save best checkpoint directly to weights/
    weights_dir = project_root / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_weights = project_root / "runs" / "rtdetr_ppe_run" / "weights" / "best.pt"
    
    if best_weights.exists():
        import shutil
        target_path = weights_dir / "best.pt"
        shutil.copy(best_weights, target_path)
        print(f"[+] Best weights successfully exported to: {target_path}")

if __name__ == "__main__":
    run_training()
