"""
Application Configuration Settings
-----------------------------------
Central configuration module managing model paths, thresholds, and server properties.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Application Info
APP_TITLE = "Constrained Object Detection & Reasoning API"
APP_DESCRIPTION = (
    "Production-grade RT-DETR detection service with a hand-written minimal reasoning layer "
    "for Construction PPE compliance (Hard-Hat, Safety-Vest, Person). Built for RAP Screening Round."
)
APP_VERSION = "1.0.0"

# Model & Inference Settings
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", str(BASE_DIR / "weights" / "best.pt"))
FALLBACK_MODEL = "rtdetr-l.pt"

DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", 0.40))
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.50))

# Guardrail Limits (Strictness thresholds for Part B)
GUARDRAIL_MIN_CONFIDENCE = 0.45
MIN_PIXEL_AREA_FOR_HEAD = 400.0  # 20x20 pixels minimum to reliably detect a helmet

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
