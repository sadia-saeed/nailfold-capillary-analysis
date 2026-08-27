# Nailfold Capillaroscopy Analysis

**Automated Capillary Segmentation and Quantification in Nailfold Capillaroscopy Images**

An image-analysis pipeline that detects, segments, and counts capillaries in nailfold capillaroscopy images — comparing a classical computer-vision approach against a deep-learning (YOLOv8) approach, across both normal and pathological cases.

> Developed as part of a research project at Université Bourgogne Europe.

---

## Overview

Nailfold capillaroscopy is a non-invasive technique used to examine capillaries at the nailfold, helping clinicians assess microvascular health and diagnose conditions such as systemic sclerosis and other connective tissue diseases. Manually counting and characterizing capillaries in these images is time-consuming and prone to inter-observer variability.

This project explores automating that process with two complementary approaches:

1. **Traditional image processing** — filtering, vessel enhancement, and morphological operations to segment and count capillaries.
2. **Deep learning (YOLOv8)** — a trained instance-segmentation model to detect and count capillaries with greater robustness on pathological images.

## Pipeline

```
                     Nailfold Capillaroscopy Image
                                  │
                     ROI Extraction (black line alignment)
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   Traditional Image Processing                  Deep Learning (YOLOv8)
   • CLAHE contrast enhancement                  • Manual annotation (Roboflow)
   • Hessian / vesselness filtering               • VOC → YOLO format conversion
   • Adaptive thresholding                        • Train/val dataset preparation
   • Morphological opening/closing                • YOLOv8-seg fine-tuning
   • Connected-component analysis                  • ROI-filtered inference + NMS
              │                                        │
              └───────────────────┬───────────────────┘
                                  ▼
                     Capillary Segmentation & Count
```

## Methodology

### 1. Region of Interest (ROI) Extraction
Each image contains a black reference line marking the capillary bed. The pipeline:
- Thresholds the image to isolate the darkest pixels and detects the line via contour analysis.
- Fits a line to the largest contour and computes the rotation angle needed to make it horizontal.
- Rotates the image and extracts a consistent rectangular ROI directly above the line.

### 2. Approach 1 — Traditional Image Processing
- **Contrast enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Vessel enhancement:** Hessian matrix / eigenvalue-based "vesselness" filtering to highlight tubular structures (inspired by Frangi vesselness filtering).
- **Binarization:** Adaptive Gaussian thresholding.
- **Morphological cleanup:** Opening followed by closing to remove noise and fill small gaps.
- **Connected-component analysis:** Blobs are filtered by area and eccentricity (elongated shapes only) to isolate capillary-like structures.
- **Counting & visualization:** Centroids are computed, labeled, and overlaid on the original ROI.

### 3. Approach 2 — Deep Learning (YOLOv8)
- **Annotation:** Capillaries manually labeled (polygons/bounding boxes) in Roboflow, exported as Pascal VOC XML.
- **Format conversion:** VOC XML annotations converted to normalized YOLOv8 segmentation format.
- **Dataset preparation:** Automatic train/val split with a generated `capillary_dataset.yaml`.
- **Training:** `yolov8s-seg` fine-tuned on the custom dataset (CPU-optimized: batch size 4, 100 epochs, early stopping, tailored augmentation for small objects).
- **Inference:** Detections filtered to those whose centroids fall inside the ROI, with custom non-max suppression (IoU ≤ 0.3) to remove overlapping predictions.

## Results

The two approaches were evaluated against manually annotated ground-truth capillary counts:

| Approach | Metric | Result |
|---|---|---|
| Traditional image processing | Mean Absolute Error (count) | 1.90 |
| YOLOv8 segmentation | Qualitative / mAP (dataset-dependent) | See notebook |

A bar chart comparing predicted vs. ground-truth counts across normal (N1, N2) and pathological (S2, S3) samples is included in the notebook.

**Key takeaway:** both approaches showed promise, but neither was fully reliable across the whole dataset — largely due to the limited size and consistency of manual annotations. Performance is expected to improve substantially with a larger, expert-annotated dataset (ideally reviewed by capillaroscopy specialists).

## Usage

1. Place your capillaroscopy images (and, for training, Pascal VOC XML annotations) under `DATA/DATA/train/`.
2. Open `project.ipynb` in Jupyter Lab/Notebook.
3. Run the cells in order:
   - **ROI extraction** to align and crop each image.
   - **Traditional pipeline** to segment and count capillaries classically.
   - **Dataset preparation + YOLOv8 training** to build and fine-tune the deep-learning model.
   - **Inference & visualization** to run the trained model on new images.
   - **Evaluation** to compare predicted counts against ground truth.

To run inference with the pretrained weights on a new image:

```python
from ultralytics import YOLO

model = YOLO("capillary_detection_model_S.pt")
results = model("path/to/image.jpg")
results[0].show()
```

## Limitations & Future Work

- **Annotation quality:** annotations were created manually and may lack the precision needed for optimal model training; expert-reviewed annotations would likely improve results significantly.
- **Dataset size:** a larger, more diverse dataset (covering more pathological variations) is needed for the YOLOv8 model to generalize well.
- **Robustness:** the traditional pipeline is sensitive to contrast, background noise, and morphological variation across images.
- **Next steps:** expand the annotated dataset, explore stronger backbone architectures, and validate against clinician-reviewed ground truth.

## References

- Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization (CLAHE)*.
- Frangi, A. F. et al. *Multiscale Vessel Enhancement Filtering*.
- Roboflow — annotation and dataset management platform.
- *YOLOv8-Based System for Nail Capillary Detection*, PMC (2024).
- Ultralytics YOLOv8 documentation.

## Authors & Acknowledgments

- **Author:** Sadia Saeed
- **Institution:** Université Bourgogne Europe
- **Supervisors:** Stephanie Bricq, Maya Faraji Zamharir
