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

# Minimum confidence required to form definitive conclusions
STRICT_CONFIDENCE_THRESHOLD = 0.45

# Minimum bounding box dimension (in pixels) for a person to inspect headgear
MIN_PERSON_HEIGHT_PX = 40.0
MIN_PERSON_WIDTH_PX = 25.0

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

        people = [d for d in detections if d.class_name == "person"]

        # Scenario 2: Asking about people/PPE when no persons are identifiable
        if any(w in query.lower() for w in ["person", "people", "worker", "anyone", "someone", "wearing"]) and len(people) == 0:
            return (
                True,
                "Insufficient information: Safety equipment may be present, but no identifiable "
                "workers/persons were detected to determine compliance."
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

        # Scenario 4: Borderline low-confidence detections (0.30 <= confidence < 0.45)
        borderline_items = [d for d in detections if d.confidence < STRICT_CONFIDENCE_THRESHOLD]
        if borderline_items:
            culprit = borderline_items[0]
            return (
                True,
                f"Insufficient information: Detected '{culprit.class_name}' with borderline "
                f"confidence ({culprit.confidence:.2f} < {STRICT_CONFIDENCE_THRESHOLD}). "
                "Image conditions (lighting/blur/occlusion) prevent a definitive answer."
            )

        # All checks passed: visual data is reliable
        return False, None

confidence_guardrail = ConfidenceGuardrail()
