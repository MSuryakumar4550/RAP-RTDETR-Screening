"""
Pydantic Schemas / Data Transfer Objects (DTO)
-----------------------------------------------
Defines the strict contracts for API requests and responses.
Java Equivalent: POJOs / Response DTOs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """
    Absolute pixel coordinates of the detected bounding box.
    [x1, y1] = Top-Left corner, [x2, y2] = Bottom-Right corner.
    """
    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")

class DetectionItem(BaseModel):
    """Represents a single detected object in an image."""
    class_id: int = Field(..., description="Numeric class identifier (0: hard-hat, 1: safety-vest, 2: person)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Bounding box spatial coordinates")

class DetectionResponse(BaseModel):
    """Part A API Response Schema for POST /api/v1/detect."""
    status: str = Field(default="success", description="Request status")
    inference_time_ms: float = Field(..., description="Time taken by the RT-DETR model to infer in milliseconds")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    total_detections: int = Field(..., description="Total count of objects detected above threshold")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected objects")

class ReasoningResponse(BaseModel):
    """Part B API Response Schema for POST /api/v1/reason."""
    status: str = Field(default="success", description="Request status")
    question: str = Field(..., description="The original natural-language query asked by user")
    intent: str = Field(..., description="Detected query intent (e.g., IMAGE_OBJECT_QUERY vs UNRELATED_QUERY)")
    detector_called: bool = Field(..., description="Whether the Part A RT-DETR detector was invoked")
    confidence_guardrail_triggered: bool = Field(..., description="True if results were ambiguous or low confidence")
    answer: str = Field(..., description="Plain-English natural-language answer")
    raw_detections_summary: Optional[dict] = Field(default=None, description="Object counts summary used for reasoning")
