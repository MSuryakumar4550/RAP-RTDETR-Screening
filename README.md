# Constrained Object Detection & Reasoning API (RT-DETR)

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Model](https://img.shields.io/badge/Model-RT--DETR--Large-FF6F00.svg?style=flat)](https://docs.ultralytics.com/models/rtdetr/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%7C%20Compose-2496ED.svg?style=flat&logo=Docker)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-success.svg?style=flat)]()

**Pre-Hackathon Screening Submission for Rapid Acceleration Partners (RAP)**  
**Track:** Computer Vision + Applied ML Engineering (with a light Agentic component)

---

## Executive Summary
This repository delivers an end-to-end, production-grade Computer Vision and Reasoning microservice built on **RT-DETR (Real-Time Detection Transformer)** fine-tuned for **Construction Site Safety & Worker PPE Compliance Monitoring**.

### Key Highlights & Compliance with RAP Constraints:
1. **At Least One Non-COCO Class (Constraint #3)**: Models **`hard-hat`** (Class 0) and **`safety-vest`** (Class 1) alongside **`person`** (Class 2). Submissions with standard COCO classes receive zero.
2. **Zero Agentic Frameworks (Constraint #1)**: Built **strictly without LangChain, LangGraph, CrewAI, AutoGen**. The reasoning engine is an explicit, pure-Python state machine with deterministic Intent Routing, 2D Spatial Containment, and Confidence Guardrails.
3. **No AutoML / Proprietary Training (Constraint #2)**: Trained using an open, auditable PyTorch pipeline with Ultralytics RT-DETR.
4. **Reproducibility Locked (Constraint #4)**: Fixed global seeds (`seed=42`), deterministic CUDA configurations, and explicit hyperparameter manifests.
5. **Production Dockerization (+10% Bonus)**: Single-command container deployment via `docker-compose up`.

---

## System Architecture

```
User Request (Image + Query)
             │
             ▼
   [POST /api/v1/reason]
             │
             ├──> [1. Intent Router (router.py)]
             │          ├──> Unrelated Query -> Instant response (0 GPU cycles)
             │          └──> Inspection Query -> Invoke Detector
             │
             ├──> [2. Part A RT-DETR Model (detector.py)]
             │          └──> Generates Bounding Boxes, Confidences, Classes
             │
             ├──> [3. Confidence Guardrail (guardrails.py)]
             │          ├──> Low Conf (<0.45) / Distant (<40px) -> "Insufficient Information"
             │          └──> Sufficient Visual Evidence -> Proceed to Spatial Reasoning
             │
             └──> [4. Spatial Reasoning Engine (spatial_engine.py)]
                        └──> Bounding Box Containment: Head (Top 25%) & Torso (20-65%)
                        └──> Final Structured Plain-English Response
```

---

## Directory Structure
```
rap-rtdetr-screening/
├── app/
│   ├── core/
│   │   ├── config.py             # Model paths, thresholds, server parameters
│   │   └── detector.py           # Singleton RT-DETR inference service (<30ms)
│   ├── reasoning/
│   │   ├── router.py             # Pure-Python Intent Router (Visual vs Unrelated)
│   │   ├── guardrails.py         # Confidence & Ambiguity Guardrails
│   │   └── spatial_engine.py     # 2D Geometry spatial containment reasoning
│   ├── schemas/
│   │   └── models.py             # Pydantic DTOs for strict request/response validation
│   └── main.py                   # FastAPI application routes (/detect, /reason, /health)
├── dataset/
│   ├── data.yaml                 # Ultralytics dataset specification
│   ├── prepare_dataset.py        # Dataset validation and split audit script
│   ├── download_dataset.py       # Automated Roboflow / Colab download utility
│   └── README.md                 # 70/15/15 split justification against temporal leakage
├── training/
│   ├── hyperparams.yaml          # Reproducible hyperparameter configuration
│   ├── train.py                  # Standalone training script with locked seed=42
│   ├── evaluate.py               # Benchmark script (mAP@50, mAP@50:95, confusion matrix)
│   └── RT_DETR_PPE_Training.ipynb# Google Colab GPU-accelerated training notebook
├── weights/
│   └── README.md                 # Direct download links for best.pt weights
├── memo/
│   ├── TECHNICAL_MEMO.md         # 2-Page Engineering Report (Rubric Sections 1-6)
│   └── failure_cases_analysis.md # Deep root-cause analysis on 5 real failure modes
├── tests/
│   └── test_api.py               # Automated pytest suite (Router, Spatial Math, Guardrails)
├── Dockerfile                    # Containerization manifest
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Pinned production dependencies
└── README.md                     # This documentation
```

---

## Quickstart & Deployment

### Option A: Local Python Environment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: **`http://localhost:8000/docs`**

### Option B: Docker Container Deployment (Recommended)
```bash
# Build and launch container in detached mode
docker-compose up --build -d

# Check service health
curl http://localhost:8000/health
```

---

## API Endpoints & Usage

### 1. Part A: Object Detection (`POST /api/v1/detect`)
Accepts an image and returns detected objects, bounding boxes, and confidence scores.

**cURL Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect?confidence_threshold=0.40" \
     -H "accept: application/json" \
     -F "file=@sample_worker.jpg"
```

**JSON Response:**
```json
{
  "status": "success",
  "inference_time_ms": 28.4,
  "image_width": 1280,
  "image_height": 720,
  "total_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "hard-hat",
      "confidence": 0.912,
      "bbox": {"x1": 450.2, "y1": 120.5, "x2": 512.0, "y2": 185.0, "width": 61.8, "height": 64.5}
    },
    {
      "class_id": 2,
      "class_name": "person",
      "confidence": 0.941,
      "bbox": {"x1": 420.0, "y1": 115.0, "x2": 580.0, "y2": 580.0, "width": 160.0, "height": 465.0}
    }
  ]
}
```

---

### 2. Part B: Natural-Language Reasoning (`POST /api/v1/reason`)
Accepts an image and a natural-language question. Demonstrates **Intent Routing**, **Structured Spatial Reasoning**, and **Confidence Guardrails**.

#### Example 1: Compliance Query (Visual Inspection)
```bash
curl -X POST "http://localhost:8000/api/v1/reason" \
     -F "file=@sample_worker.jpg" \
     -F "question=Is anyone not wearing a helmet?"
```
**Response:**
```json
{
  "status": "success",
  "question": "Is anyone not wearing a helmet?",
  "intent": "IMAGE_OBJECT_QUERY",
  "detector_called": true,
  "confidence_guardrail_triggered": false,
  "answer": "Yes, all 1 worker(s) in this image are wearing safety helmets.",
  "raw_detections_summary": {
    "total_workers": 1,
    "total_helmets": 1,
    "total_vests": 1,
    "compliant_workers": 1
  }
}
```

#### Example 2: Unrelated Query (Bypasses Detector)
```bash
curl -X POST "http://localhost:8000/api/v1/reason" \
     -F "file=@sample_worker.jpg" \
     -F "question=What is the capital of France?"
```
**Response:**
```json
{
  "status": "success",
  "question": "What is the capital of France?",
  "intent": "UNRELATED_QUERY",
  "detector_called": false,
  "confidence_guardrail_triggered": false,
  "answer": "The question 'What is the capital of France?' does not appear to be related to the visual contents or safety compliance of the uploaded image. The detection pipeline was not invoked."
}
```

#### Example 3: Confidence Guardrail ("Insufficient Information")
When a worker is $>50$m in the background ($<30$px height) or confidence is borderline ($0.35$):
```json
{
  "status": "success",
  "question": "Is anyone not wearing a helmet?",
  "intent": "IMAGE_OBJECT_QUERY",
  "detector_called": true,
  "confidence_guardrail_triggered": true,
  "answer": "Insufficient information: Worker #1 is located too far in the background (22x36px). Resolution is too low to reliably verify whether a hard-hat or safety vest is worn."
}
```

---

## Running Automated Tests
To run the end-to-end test suite validating schema adherence, intent classification, and spatial geometry:
```bash
pytest tests/ -v
```

---

## Submission Deliverables Index
- **Source Code**: This repository (Training script, evaluation script, FastAPI application, test suite).
- **Model Checkpoints**: Download paths documented in [`weights/README.md`](weights/README.md).
- **2-Page Technical Memo**: Formatted in [`memo/TECHNICAL_MEMO.md`](memo/TECHNICAL_MEMO.md).
- **5 Failure Case Root-Cause Analysis**: Documented in [`memo/failure_cases_analysis.md`](memo/failure_cases_analysis.md).
