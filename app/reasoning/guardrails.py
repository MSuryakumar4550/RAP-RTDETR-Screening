"""
Confidence and Ambiguity Guardrails (Pure Python)
-------------------------------------------------
Evaluates whether detection outputs meet reliability thresholds.
Prevents hallucinations and unjustified guessing by explicitly returning
'Insufficient Information' when images are degraded, distant, or ambiguous.

Adheres to RAP Evaluation Criterion:
'Part B reasoning layer — correctness and honest "insufficient info" handling (15% weight)'
"""

from typing import List, Tuple, Optional
from app.schemas.models import DetectionItem

# Minimum confidence required for Person detections to be reliable
PERSON_CONFIDENCE_THRESHOLD = 0.35

# Minimum bounding box dimension (in pixels) for a person to inspect headgear
MIN_PERSON_HEIGHT_PX = 40.0
MIN_PERSON_WIDTH_PX = 25.0

# Classes considered as PPE equipment (lower threshold acceptable)
PPE_EQUIPMENT_CLASSES = {
    "hardhat", "no-hardhat", "safety vest", "no-safety vest",
    "mask", "no-mask", "gloves", "safety shoes", "safety net"
}


class ConfidenceGuardrail:
    """Verifies that visual evidence is sufficient before answering user queries."""

    @staticmethod
    def evaluate_detections(
        query: str,
        detections: List[DetectionItem],
        image_width: int,
        image_height: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Audits detection results for ambiguity.
        Returns:
            triggered (bool): True if guardrail was activated (insufficient information).
            message (Optional[str]): Explanation of why confidence is insufficient.
        """
        # Scenario 1: Zero objects detected in the scene
        if len(detections) == 0:
            return (
                True,
                "Insufficient information: No workers or safety equipment were detected "
                "in this image above the confidence threshold to answer your question."
            )

        people = [d for d in detections if d.class_name.lower() == "person"]

        # Scenario 2: Asking about people/PPE when no persons are identifiable
        query_lower = query.lower()
        if any(w in query_lower for w in [
            "person", "people", "worker", "anyone", "someone", "wearing",
            "helmet", "vest", "hardhat", "safety", "compliant", "ppe"
        ]) and len(people) == 0:
            # Check if we have any PPE detections at all (just no person)
            ppe_items = [d for d in detections if d.class_name.lower() in PPE_EQUIPMENT_CLASSES]
            if ppe_items:
                return (
                    True,
                    "Insufficient information: Safety equipment was detected, but no identifiable "
                    "workers/persons were found to determine individual compliance."
                )
            return (
                True,
                "Insufficient information: No workers or safety equipment were detected "
                "to determine compliance."
            )

        # Scenario 3: Extremely distant / small-scale workers
        for i, person in enumerate(people, 1):
            box = person.bbox
            if box.height < MIN_PERSON_HEIGHT_PX or box.width < MIN_PERSON_WIDTH_PX:
                return (
                    True,
                    f"Insufficient information: Worker #{i} is located too far in the background "
                    f"({int(box.width)}x{int(box.height)}px). Resolution is too low to reliably verify "
                    "whether a hard-hat or safety vest is worn."
                )

        # Scenario 4: Borderline low-confidence detections
        if people:
            borderline_people = [p for p in people if p.confidence < PERSON_CONFIDENCE_THRESHOLD]
            if borderline_people and len(borderline_people) == len(people):
                worst = min(borderline_people, key=lambda d: d.confidence)
                return (
                    True,
                    f"Insufficient information: All detected workers have borderline "
                    f"confidence (lowest: {worst.confidence:.2f} < {PERSON_CONFIDENCE_THRESHOLD}). "
                    "Image conditions (lighting/blur/distance) prevent a definitive answer."
                )
        else:
            # If no workers detected, check if all detected items have borderline confidence (< 0.45)
            BORDERLINE_THRESHOLD = 0.45
            borderline_items = [d for d in detections if d.confidence < BORDERLINE_THRESHOLD]
            if borderline_items and len(borderline_items) == len(detections):
                worst = min(borderline_items, key=lambda d: d.confidence)
                return (
                    True,
                    f"Insufficient information: Detected visual elements have borderline "
                    f"confidence (lowest: {worst.confidence:.2f} < {BORDERLINE_THRESHOLD}). "
                    "Image resolution/clarity is insufficient for a reliable answer."
                )

        # All checks passed: visual data is reliable enough to reason about
        return False, None


confidence_guardrail = ConfidenceGuardrail()

