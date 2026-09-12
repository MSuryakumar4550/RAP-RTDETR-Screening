# Dataset Specification: Construction Site Safety & Worker PPE

## 1. Domain Justification
In automated enterprise visual inspection, workplace safety compliance (OSHA / Directorate General of Mines Safety regulations) is a high-impact application. Traditional manual audits are slow and sporadic. An autonomous Computer Vision and Reasoning system monitoring continuous video streams can instantly flag workers without safety equipment (Hard-Hats, High-Visibility Vests).

## 2. Class Definitions
The dataset is annotated with 3 classes:
- **`hard-hat` (Class 0 - Non-COCO)**: Safety helmets worn by construction workers, engineers, and visitors.
- **`safety-vest` (Class 1 - Non-COCO)**: High-visibility fluorescent vests (reflective yellow, orange, neon green).
- **`person` (Class 2 - Reference Class)**: The human body bounding box used for spatial anchor reasoning.

> **Constraint Compliance**: Both `hard-hat` and `safety-vest` are non-standard COCO classes, strictly adhering to RAP Hard Constraint #3.

---

## 3. Train / Validation / Test Split Strategy & Technical Justification

The dataset is partitioned into an **approximate 70% / 15% / 15% stratified split**:
- **Train Set (~70%)**: Optimizes model weights (backbone attention and bipartite prediction heads).
- **Validation Set (~15%)**: Evaluated at the end of each epoch to track validation loss, compute mAP@50, and trigger early stopping / model checkpoint saving (`best.pt`).
- **Test Set (~15%)**: A completely pristine hold-out set never seen during training or hyperparameter tuning, evaluated solely for the final evaluation report.

### Technical Justification for Stratified Partitioning:
1. **Preventing Data Leakage**: Video frames extracted from continuous CCTV streams can suffer from severe temporal correlation (adjacent frames are almost identical). The split was performed by **scene / video sequence grouping** rather than random frame shuffle, ensuring frames from the same camera angle/time window do not appear in both train and test splits.
2. **Class Imbalance Preservation**: Construction scenes often contain more vests and helmets than unequipped individuals. Stratified sampling preserves natural class co-occurrence frequencies across all three partitions.

---

## 4. Annotation Format
Standard YOLO/RT-DETR text format:
Each image `image_name.jpg` has a corresponding `image_name.txt` file in the `labels/` directory.
Each row consists of:
`<class_id> <x_center> <y_center> <width> <height>` (normalized to `[0.0, 1.0]`).
