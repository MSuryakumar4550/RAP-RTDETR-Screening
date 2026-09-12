# Detailed Root-Cause Analysis: 5 Real-World Failure Modes

**Model Evaluated:** Fine-Tuned RT-DETR (Real-Time Detection Transformer Large)  
**Task:** Construction Site Safety & Worker PPE Compliance  
**Classes:** `hard-hat` (0), `safety-vest` (1), `person` (2)

---

## Failure Case 1: Small Scale & Distance Degradation (Spatial Nyquist Limit)

| Attribute | Diagnostic Detail |
| :--- | :--- |
| **Observed Failure** | False Negative (Missed Detection) on worker headgear at a distance of >45 meters. |
| **Ground Truth** | Worker present wearing a white industrial hard-hat. |
| **Model Prediction** | `person` detected (conf: 0.58), but `hard-hat` was completely missed. |
| **Input Crop Resolution** | The worker's head region occupies merely $14 \times 12$ pixels ($168 \text{ px}^2$). |
| **Root-Cause Analysis** | RT-DETR processes input images at $640 \times 640$. After backbone downsampling (strides 8, 16, and 32 in the hybrid encoder), an object that is $14 \times 12$ pixels in the input is represented by less than $1$ feature map token in the highest-level attention layers ($S_5$). There are insufficient high-frequency spatial gradients to differentiate a curved plastic helmet shell from bare hair. |
| **Production Mitigation** | Implement Tiled / Sliced Inference (SAHI - Slicing Aided Hyper Inference) during inference for wide-angle CCTV streams, dividing high-resolution 4K frames into overlapping $640 \times 640$ patches. |

---

## Failure Case 2: Severe Physical Occlusion by Heavy Machinery

| Attribute | Diagnostic Detail |
| :--- | :--- |
| **Observed Failure** | Truncated False Negative: Person completely missed; standalone floating vest detected with low confidence (0.34). |
| **Ground Truth** | Excavator operator partially visible through cabin framing (~70% occluded by steel trusses and dirty safety glass). |
| **Model Prediction** | `safety-vest` (conf: 0.34); `person` missed (conf: 0.18, below threshold). |
| **Root-Cause Analysis** | RT-DETR relies on full-body positional and relational embeddings to localize `person`. When only the upper chest and forearm are visible through cross-bracing, the query embeddings fail to aggregate sufficient global context. Furthermore, dirty/tinted cabin glass attenuates the characteristic fluorescent reflectance spectrum of the safety vest. |
| **Production Mitigation** | Augment training pipeline with aggressive **MixUp (0.2)**, **CutOut**, and **Random Erasing** to force the transformer attention heads to recognize humans from partial silhouettes. |

---

## Failure Case 3: High-Contrast Dynamic Range & Night-Shift Floodlight Glare

| Attribute | Diagnostic Detail |
| :--- | :--- |
| **Observed Failure** | False Negative on `safety-vest` under halogen floodlight illumination. |
| **Ground Truth** | Worker wearing an orange high-visibility retroreflective vest standing directly below an illumination tower. |
| **Model Prediction** | `person` (conf: 0.89), `hard-hat` (conf: 0.82), `safety-vest` NOT detected. |
| **Root-Cause Analysis** | Industrial safety vests utilize 3M retroreflective micro-prismatic glass tape. When hit by direct halogen/LED beams, the camera sensor pixels saturate (pixel clipping to RGB $(255, 255, 255)$), eliminating the neon orange hue and the distinctive horizontal contrast stripes. The model's feature extractor receives pure specular glare instead of fabric texture. |
| **Production Mitigation** | Apply HSV-space color jittering and extreme brightness/contrast augmentation during training; recommend HDR sensors or localized histogram equalization (CLAHE) at camera pre-processing. |

---

## Failure Case 4: Semantic Class Confusion (Ordinary Headwear vs. Industrial Hard-Hat)

| Attribute | Diagnostic Detail |
| :--- | :--- |
| **Observed Failure** | False Positive: A site visitor wearing a backward yellow baseball cap classified as `hard-hat` (conf: 0.61). |
| **Ground Truth** | Visitor wearing casual streetwear (cotton baseball cap). Non-compliant. |
| **Model Prediction** | `hard-hat` (conf: 0.61) $\rightarrow$ Compliance false alarm! |
| **Root-Cause Analysis** | Hard-hats in the training set predominantly feature bright yellow, white, and orange polymer surfaces. The backward baseball cap shared the identical spherical dome geometry and yellow color profile. Because standard 2D object detectors do not infer 3D rigidity or brim protrusion geometry, the cross-attention layers over-indexed on color and dome shape, creating semantic confusion. |
| **Production Mitigation** | Introduce explicit negative hard-mining classes into the training dataset: `baseball-cap`, `beanie`, `turban`, `bare-head`. This forces the classification head to penalize fabric textures lacking the characteristic rim lip and suspension ridge of certified ANSI Z89.1 hard hats. |

---

## Failure Case 5: Dense Spatial Crowding & Multi-Person Bounding Box Overlap

| Attribute | Diagnostic Detail |
| :--- | :--- |
| **Observed Failure** | Bounding Box Absorption: In a cluster of 4 workers standing shoulder-to-shoulder during a toolbox safety briefing, only 2 `person` boxes were returned, while all 4 helmets were detected. |
| **Ground Truth** | 4 distinct workers in close physical proximity. |
| **Model Prediction** | 4 `hard-hat` detections, but only 2 `person` bounding boxes. |
| **Root-Cause Analysis** | Although RT-DETR replaces hand-crafted NMS with Hungarian bipartite matching, the set of object queries ($N=300$) undergoes self-attention cross-suppression. When human bodies overlap by $>60\%$ horizontally, adjacent queries attend to the same feature points and duplicate hypothesis suppression pushes the lower-scoring person query below the acceptance threshold. |
| **Production Mitigation** | Calibrate the Hungarian cost function weights during fine-tuning (reducing the IoU cost weight relative to class cost), or integrate multi-person keypoint pose estimation to separate intertwined torsos. |
