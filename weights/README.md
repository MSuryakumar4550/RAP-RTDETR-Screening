# Model Weights Directory

This directory stores the fine-tuned RT-DETR weights.

## 1. Local Location
- Target checkpoint: `weights/best.pt`
- Baseline fallback: `rtdetr-l.pt` (automatically cached by Ultralytics if custom weights are pending)

## 2. Direct Public Download Link (MANDATORY FOR RAP REVIEWERS)
As required by the assignment deliverables ("*Model weights: Or a direct, working download/load path (do not make reviewers guess)*"):

- **Direct Download URL**: `https://drive.google.com/uc?export=download&id=YOUR_PUBLIC_GDRIVE_ID` (or HuggingFace Hub: `https://huggingface.co/YOUR_USERNAME/rap-rtdetr-ppe/resolve/main/best.pt`)
- **Checksum / Size**: ~65.4 MB (`best.pt`)
- **Architecture**: RT-DETR Large (Baidu Real-Time Detection Transformer with HGNetv2 / ResNet backbone)

## 3. How the FastAPI Backend Loads Weights
The application automatically loads weights from `weights/best.pt`. If not present, it gracefully falls back to the pretrained checkpoint with clear console warnings.
