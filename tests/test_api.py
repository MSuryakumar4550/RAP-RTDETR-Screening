"""
Automated Unit & Integration Test Suite
----------------------------------------
Validates both FastAPI endpoints, intent router, spatial reasoning math,
and guardrail boundary checks.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reasoning.router import intent_router
from app.reasoning.guardrails import confidence_guardrail
from app.reasoning.spatial_engine import spatial_reasoning_engine
from app.schemas.models import DetectionItem, BoundingBox

client = TestClient(app)

def create_test_image_bytes(width: int = 640, height: int = 640, color=(128, 128, 128)) -> bytes:
    """Helper to generate in-memory dummy JPEG images for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ==========================================
# 1. Health & Validation Tests
# ==========================================
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_detect_invalid_file_type():
    """Verifies that non-image payloads are rejected with 400 Bad Request."""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]

# ==========================================
# 2. Part B: Intent Router Unit Tests
# ==========================================
def test_intent_router_visual_queries():
    visual_queries = [
        "How many people are in this image?",
        "Is anyone not wearing a helmet?",
        "What's the most common object here?",
        "Are all workers wearing high visibility vests?"
    ]
    for q in visual_queries:
        intent, requires_det = intent_router.classify_intent(q)
        assert intent == "IMAGE_OBJECT_QUERY"
        assert requires_det is True

def test_intent_router_unrelated_queries():
    unrelated_queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "Solve 25 * 4",
        "Explain quantum computing"
    ]
    for q in unrelated_queries:
        intent, requires_det = intent_router.classify_intent(q)
        assert intent == "UNRELATED_QUERY"
        assert requires_det is False

def test_part_b_unrelated_query_bypasses_detector():
    """Ensures unrelated queries consume 0 GPU cycles without running detector."""
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/v1/reason",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"question": "What is the capital of Australia?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "UNRELATED_QUERY"
    assert data["detector_called"] is False
    assert data["confidence_guardrail_triggered"] is False
    assert "does not appear to be related" in data["answer"]

# ==========================================
# 3. Part B: Guardrail Boundary Tests
# ==========================================
def test_guardrail_triggers_on_distant_small_worker():
    """Verifies guardrail triggers when worker is too far in background (<40px height)."""
    tiny_worker = [
        DetectionItem(
            class_id=2,
            class_name="person",
            confidence=0.85,
            bbox=BoundingBox(x1=100, y1=100, x2=120, y2=130, width=20, height=30)  # height < 40px
        )
    ]
    triggered, msg = confidence_guardrail.evaluate_detections(
        query="Is he wearing a helmet?",
        detections=tiny_worker,
        image_width=1280,
        image_height=720
    )
    assert triggered is True
    assert "Resolution is too low" in msg

def test_guardrail_triggers_on_borderline_confidence():
    """Verifies guardrail triggers on borderline confidence (0.35 < 0.45)."""
    low_conf_item = [
        DetectionItem(
            class_id=0,
            class_name="hard-hat",
            confidence=0.35,
            bbox=BoundingBox(x1=100, y1=50, x2=160, y2=100, width=60, height=50)
        )
    ]
    triggered, msg = confidence_guardrail.evaluate_detections(
        query="What equipment is here?",
        detections=low_conf_item,
        image_width=640,
        image_height=640
    )
    assert triggered is True
    assert "borderline confidence" in msg

# ==========================================
# 4. Part B: Spatial Reasoning Math Tests
# ==========================================
def test_spatial_reasoning_compliance():
    """
    Tests that a hard-hat located on top of a person's head
    is correctly recognized as compliant.
    """
    detections = [
        DetectionItem(
            class_id=2,
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400, width=100, height=300)
        ),
        DetectionItem(
            class_id=0,
            class_name="hard-hat",
            confidence=0.88,
            bbox=BoundingBox(x1=220, y1=95, x2=280, y2=150, width=60, height=55)  # sits on top of head
        ),
        DetectionItem(
            class_id=1,
            class_name="safety-vest",
            confidence=0.90,
            bbox=BoundingBox(x1=210, y1=170, x2=290, y2=280, width=80, height=110) # sits on torso
        )
    ]

    answer = spatial_reasoning_engine.answer_query("Is anyone not wearing a helmet?", detections)
    assert "all 1 worker(s) in this image are wearing safety helmets" in answer

    count_answer = spatial_reasoning_engine.answer_query("How many workers are present?", detections)
    assert "1 worker detected" in count_answer
