"""
RT-DETR Detector Service Layer
------------------------------
Singleton inference service that loads the model weights once and executes
low-latency computer vision predictions on incoming image payloads.
Java Equivalent: DetectionService (Spring @Service / Singleton Bean).
"""

import io
import time
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
from ultralytics import RTDETR

from app.core import config
from app.schemas.models import BoundingBox, DetectionItem, DetectionResponse

logger = logging.getLogger("detector-service")

CLASS_NAMES = {
    0: "hard-hat",
    1: "safety-vest",
    2: "person"
}

class RTDETRDetector:
    """Singleton detector instance to avoid reloading heavy weights per request."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RTDETRDetector, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        weights_file = Path(config.WEIGHTS_PATH)
        if weights_file.exists():
            logger.info(f"Loading custom fine-tuned weights from: {weights_file}")
            self.model = RTDETR(str(weights_file))
        else:
            logger.warning(
                f"Custom weights not found at {weights_file}. "
                f"Falling back to pretrained backbone '{config.FALLBACK_MODEL}'..."
            )
            self.model = RTDETR(config.FALLBACK_MODEL)
        
        # Warm-up inference on a dummy 640x640 blank image to optimize GPU cache
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        logger.info("RT-DETR detector successfully initialized and warmed up.")

    def detect_image(
        self, 
        image_bytes: bytes, 
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD
    ) -> DetectionResponse:
        """
        Processes raw image bytes and returns structured detections.
        """
        # 1. Parse Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

        img_width, img_height = image.size

        # 2. Run Inference & Measure Latency
        start_time = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Post-Process Bounding Boxes
        detection_items: List[DetectionItem] = []
        result = results[0]  # First image in batch

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()     # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(float, box)
                w = x2 - x1
                h = y2 - y1

                # Resolving class name
                name = CLASS_NAMES.get(cls_id, result.names.get(cls_id, f"class_{cls_id}"))

                item = DetectionItem(
                    class_id=cls_id,
                    class_name=name,
                    confidence=round(float(conf), 4),
                    bbox=BoundingBox(
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2),
                        width=round(w, 2),
                        height=round(h, 2)
                    )
                )
                detection_items.append(item)

        return DetectionResponse(
            status="success",
            inference_time_ms=round(inference_time_ms, 2),
            image_width=img_width,
            image_height=img_height,
            total_detections=len(detection_items),
            detections=detection_items
        )

# Global singleton accessor
detector_service = RTDETRDetector()
