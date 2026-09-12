"""
FastAPI Main Application Entrypoint
-----------------------------------
Exposes:
1. POST /api/v1/detect  - Part A: Object detection endpoint (returns classes, boxes, confidences)
2. POST /api/v1/reason  - Part B: Minimal hand-written reasoning layer (built in Step 4)
3. GET /health          - Service health probe
"""

import logging
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core import config
from app.core.detector import detector_service
from app.reasoning.router import intent_router
from app.reasoning.guardrails import confidence_guardrail
from app.reasoning.spatial_engine import spatial_reasoning_engine
from app.schemas.models import DetectionResponse, ReasoningResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api-main")

app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming API status and model readiness."""
    return {
        "status": "healthy",
        "service": config.APP_TITLE,
        "version": config.APP_VERSION,
        "model_loaded": detector_service.model is not None
    }

@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Part A - Detection"],
    summary="Run RT-DETR object detection on an image",
    response_description="Returns detected objects with bounding boxes and confidence scores"
)
async def detect_objects(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    confidence_threshold: float = Query(
        default=config.DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.05,
        le=1.0,
        description="Minimum confidence score threshold"
    )
):
    """
    Part A Core Endpoint:
    Accepts an image and returns detected objects, bounding boxes, and confidence scores.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: '{file.content_type}'. Allowed types: {list(ALLOWED_CONTENT_TYPES)}"
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received."
            )

        response = detector_service.detect_image(
            image_bytes=contents,
            conf_threshold=confidence_threshold
        )
        return response

    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        logger.error(f"Inference error encountered: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(exc)}"
        )

@app.post(
    "/api/v1/reason",
    response_model=ReasoningResponse,
    status_code=status.HTTP_200_OK,
    tags=["Part B - Reasoning Layer"],
    summary="Ask a natural-language question about an image (Zero frameworks)",
    response_description="Returns intent routing, spatial reasoning, and guardrail decisions"
)
async def reason_about_image(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    question: str = Form(..., description="Natural language question (e.g. 'Is anyone not wearing a helmet?')")
):
    """
    Part B Core Endpoint:
    Accepts an image and a natural-language question, executes pure-Python intent routing,
    structured spatial reasoning, and honest confidence guardrails.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: '{file.content_type}'"
        )

    # Step 1: Intent Routing
    intent, requires_detector = intent_router.classify_intent(question)
    logger.info(f"Query: '{question}' -> Intent: {intent} (Requires Detector: {requires_detector})")

    if not requires_detector:
        return ReasoningResponse(
            status="success",
            question=question,
            intent=intent,
            detector_called=False,
            confidence_guardrail_triggered=False,
            answer=intent_router.answer_unrelated_query(question),
            raw_detections_summary=None
        )

    # Step 2: Computer Vision Inference (Part A)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image received.")

    detection_result = detector_service.detect_image(
        image_bytes=contents,
        conf_threshold=0.30  # Low threshold so guardrails can inspect borderline confidence
    )

    # Step 3: Confidence Guardrails (Checking for Ambiguity, Distance, Low Confidence)
    is_ambiguous, guardrail_msg = confidence_guardrail.evaluate_detections(
        query=question,
        detections=detection_result.detections,
        image_width=detection_result.image_width,
        image_height=detection_result.image_height
    )

    if is_ambiguous:
        logger.warning(f"Confidence guardrail triggered for query: '{question}' -> {guardrail_msg}")
        return ReasoningResponse(
            status="success",
            question=question,
            intent=intent,
            detector_called=True,
            confidence_guardrail_triggered=True,
            answer=guardrail_msg,
            raw_detections_summary={"detected_items_count": len(detection_result.detections)}
        )

    # Step 4: Structured Spatial Reasoning
    final_answer = spatial_reasoning_engine.answer_query(
        query=question,
        detections=detection_result.detections
    )

    scene_analysis = spatial_reasoning_engine.analyze_scene(detection_result.detections)

    return ReasoningResponse(
        status="success",
        question=question,
        intent=intent,
        detector_called=True,
        confidence_guardrail_triggered=False,
        answer=final_answer,
        raw_detections_summary={
            "total_workers": scene_analysis["total_people"],
            "total_helmets": scene_analysis["total_helmets"],
            "total_vests": scene_analysis["total_vests"],
            "compliant_workers": sum(1 for w in scene_analysis["worker_profiles"] if w["compliant"])
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)
