"""
RT-DETR Detector Service Layer (Dual-Model Architecture)
---------------------------------------------------------
Uses TWO models in parallel:
1. Custom fine-tuned model (best.pt) — Detects PPE-specific classes
   (Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest, Person, etc.)
2. Base RT-DETR-L (COCO pretrained) — Reliable Person detection as backup

This dual-model approach ensures:
- Person detection never fails (COCO model is extremely reliable at 'person')
- PPE-specific detections come from the custom model
- Results are merged with deduplication via IoU-based matching

Java Equivalent: DetectionService (Spring @Service / Singleton Bean).
"""

import io
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from ultralytics import RTDETR

from app.core import config
from app.schemas.models import BoundingBox, DetectionItem, DetectionResponse

logger = logging.getLogger("detector-service")

# PPE-relevant classes we care about from the custom model
PPE_RELEVANT_CLASSES = {
    "hardhat", "no-hardhat", "safety vest", "no-safety vest",
    "person", "mask", "no-mask", "gloves", "safety shoes", "safety net"
}

# COCO class mapping (only person is relevant for PPE compliance)
COCO_PERSON_CLASS_ID = 0  # 'person' is class 0 in COCO


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two boxes (x1,y1,x2,y2)."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class RTDETRDetector:
    """Dual-model detector: Custom PPE model + COCO backbone for reliable person detection."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RTDETRDetector, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Load both the custom PPE model and the COCO base model."""
        
        # Model 1: Custom fine-tuned PPE model
        weights_file = Path(config.WEIGHTS_PATH)
        if weights_file.exists():
            logger.info(f"Loading custom fine-tuned weights from: {weights_file}")
            self.custom_model = RTDETR(str(weights_file))
            self.custom_model_available = True
            logger.info(f"Custom model classes: {self.custom_model.names}")
        else:
            logger.warning(f"Custom weights not found at {weights_file}")
            self.custom_model = None
            self.custom_model_available = False
        
        # Model 2: Base COCO pretrained model (always reliable for 'person')
        logger.info(f"Loading COCO base model: {config.FALLBACK_MODEL}")
        self.coco_model = RTDETR(config.FALLBACK_MODEL)
        
        # Expose the primary model for external checks (e.g., health endpoint)
        self.model = self.custom_model if self.custom_model_available else self.coco_model
        
        # Warm-up both models
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        if self.custom_model_available:
            self.custom_model.predict(dummy_img, verbose=False)
        self.coco_model.predict(dummy_img, verbose=False)
        
        logger.info(
            f"RT-DETR detector initialized. "
            f"Custom model: {'LOADED' if self.custom_model_available else 'NOT FOUND'}. "
            f"COCO backup: LOADED"
        )

    def _apply_class_nms(self, items: List[DetectionItem], nms_iou: float = 0.40) -> List[DetectionItem]:
        """
        Apply per-class Non-Maximum Suppression to eliminate duplicate boxes.
        The custom model often produces near-identical overlapping boxes for the
        same object. This keeps only the highest-confidence detection per cluster.
        """
        if len(items) <= 1:
            return items
        
        # Group by class
        by_class = {}
        for item in items:
            by_class.setdefault(item.class_name, []).append(item)
        
        result = []
        for cls_name, cls_items in by_class.items():
            # Sort by confidence descending
            cls_items.sort(key=lambda d: d.confidence, reverse=True)
            
            kept = []
            suppressed = set()
            
            for i, item_i in enumerate(cls_items):
                if i in suppressed:
                    continue
                kept.append(item_i)
                
                box_i = (item_i.bbox.x1, item_i.bbox.y1, item_i.bbox.x2, item_i.bbox.y2)
                for j in range(i + 1, len(cls_items)):
                    if j in suppressed:
                        continue
                    box_j = (cls_items[j].bbox.x1, cls_items[j].bbox.y1,
                             cls_items[j].bbox.x2, cls_items[j].bbox.y2)
                    if compute_iou(box_i, box_j) > nms_iou:
                        suppressed.add(j)
            
            result.extend(kept)
        
        return result

    def _run_custom_model(
        self, image: Image.Image, conf: float, iou: float
    ) -> List[DetectionItem]:
        """Run inference with the custom PPE model, with per-class NMS deduplication."""
        if not self.custom_model_available:
            return []
        
        results = self.custom_model.predict(
            source=image, conf=conf, iou=iou, imgsz=640, verbose=False
        )
        result = results[0]
        items = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # HOTFIX: The Roboflow dataset mapped the classes in an alphabetical order 
            # internally which swapped our intended labels. This dynamically corrects them.
            # FINAL DECODED MAPPING:
            # ID 0 ('hardhat') -> Finds Bare Heads (NO-Hardhat)
            # ID 1 ('no-hardhat') -> Finds Yellow Hats (Hardhat)
            # ID 2 ('no-safety vest') -> Finds Full Bodies (Person)
            # ID 3 ('person') -> Finds Vests (Safety Vest)
            # ID 4 ('safety vest') -> Finds Bodies without Vests (NO-Safety Vest)
            LABEL_REMAP = {
                'hardhat': 'NO-Hardhat',             # ID 0 -> actually NO-Hardhat
                'no-hardhat': 'Hardhat',             # ID 1 -> actually Hardhat
                'no-safety vest': 'Person',          # ID 2 -> actually Person
                'person': 'Safety Vest',             # ID 3 -> actually Safety Vest
                'safety vest': 'NO-Safety Vest'      # ID 4 -> actually NO-Safety Vest
            }

            for box, c, cls_id in zip(boxes, confidences, class_ids):
                raw_name = result.names.get(cls_id, f"class_{cls_id}")
                
                # Apply remap if it exists in the dictionary, otherwise keep original
                name = LABEL_REMAP.get(raw_name.lower(), raw_name)
                
                # Only keep PPE-relevant classes
                if name.lower() not in PPE_RELEVANT_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(float, box)
                items.append(DetectionItem(
                    class_id=cls_id,
                    class_name=name,
                    confidence=round(float(c), 4),
                    bbox=BoundingBox(
                        x1=round(x1, 2), y1=round(y1, 2),
                        x2=round(x2, 2), y2=round(y2, 2),
                        width=round(x2 - x1, 2), height=round(y2 - y1, 2)
                    )
                ))
        
        # Apply per-class NMS to remove duplicate overlapping boxes
        items = self._apply_class_nms(items, nms_iou=0.40)
        
        return items

    def _run_coco_model(
        self, image: Image.Image, conf: float, iou: float
    ) -> List[DetectionItem]:
        """Run inference with the COCO base model, extracting only 'person' detections."""
        results = self.coco_model.predict(
            source=image, conf=conf, iou=iou, imgsz=640, verbose=False
        )
        result = results[0]
        items = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, c, cls_id in zip(boxes, confidences, class_ids):
                if cls_id != COCO_PERSON_CLASS_ID:
                    continue  # Only extract 'person' from COCO
                
                x1, y1, x2, y2 = map(float, box)
                items.append(DetectionItem(
                    class_id=9,  # Map to our Person class_id (9)
                    class_name="Person",
                    confidence=round(float(c), 4),
                    bbox=BoundingBox(
                        x1=round(x1, 2), y1=round(y1, 2),
                        x2=round(x2, 2), y2=round(y2, 2),
                        width=round(x2 - x1, 2), height=round(y2 - y1, 2)
                    )
                ))
        
        return items

    def _merge_detections(
        self,
        custom_detections: List[DetectionItem],
        coco_persons: List[DetectionItem],
        iou_threshold: float = 0.50
    ) -> List[DetectionItem]:
        """
        Merge custom model and COCO model detections, deduplicating Person boxes.
        If both models detect the same person, keep the higher-confidence one.
        """
        merged = []
        
        # Add all non-Person detections from custom model (Hardhat, Vest, etc.)
        custom_persons = []
        for det in custom_detections:
            if det.class_name.lower() == "person":
                custom_persons.append(det)
            else:
                merged.append(det)
        
        # Merge Person detections: for each COCO person, check if custom model
        # already detected the same person (high IoU overlap)
        used_custom_persons = set()
        
        for coco_p in coco_persons:
            coco_box = (coco_p.bbox.x1, coco_p.bbox.y1, coco_p.bbox.x2, coco_p.bbox.y2)
            best_match_idx = -1
            best_iou = 0.0
            
            for idx, custom_p in enumerate(custom_persons):
                if idx in used_custom_persons:
                    continue
                custom_box = (custom_p.bbox.x1, custom_p.bbox.y1, 
                              custom_p.bbox.x2, custom_p.bbox.y2)
                iou = compute_iou(coco_box, custom_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_iou >= iou_threshold and best_match_idx >= 0:
                # Both models detected the same person — keep higher confidence
                used_custom_persons.add(best_match_idx)
                if coco_p.confidence >= custom_persons[best_match_idx].confidence:
                    merged.append(coco_p)
                else:
                    merged.append(custom_persons[best_match_idx])
            else:
                # COCO found a person that custom model missed
                merged.append(coco_p)
        
        # Add any custom persons that weren't matched by COCO
        for idx, custom_p in enumerate(custom_persons):
            if idx not in used_custom_persons:
                merged.append(custom_p)
        
        # Sort by confidence descending
        merged.sort(key=lambda d: d.confidence, reverse=True)
        
        return merged

    def detect_image(
        self, 
        image_bytes: bytes, 
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD
    ) -> DetectionResponse:
        """
        Processes raw image bytes using dual-model inference.
        Returns merged, deduplicated detection results.
        """
        # 1. Parse Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

        img_width, img_height = image.size

        # 2. Run Dual-Model Inference & Measure Latency
        start_time = time.perf_counter()
        
        # Run custom PPE model
        custom_detections = self._run_custom_model(image, conf_threshold, iou_threshold)
        
        # Run COCO model for reliable person detection
        coco_persons = self._run_coco_model(image, max(conf_threshold, 0.35), iou_threshold)
        
        # Merge and deduplicate
        detection_items = self._merge_detections(custom_detections, coco_persons)
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        logger.info(
            f"Dual-model inference: {len(custom_detections)} custom + "
            f"{len(coco_persons)} COCO persons -> {len(detection_items)} merged "
            f"({inference_time_ms:.0f}ms)"
        )

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
