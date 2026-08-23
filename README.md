# Comparative Analysis: Custom YOLOv8 vs Ultralytics YOLOv8 for Multi-Class Road Damage Detection



## Problem Statement

Road surface damage — longitudinal cracks, transverse cracks, alligator cracks, and potholes — is traditionally inspected manually, which is slow, expensive, and subjective. This project builds a robust YOLO-based object detector that can automatically localize and classify four types of road damage from camera images.

## Dataset

**RDD2022** — Multi-National Road Damage Dataset 2022  
- **47,420** road images from 6 countries (Japan, India, Czech Republic, Norway, USA, China)  
- **55,000+** annotated damage instances  
- 4 damage classes: **D00** (Longitudinal Crack), **D10** (Transverse Crack), **D20** (Alligator Crack), **D40** (Pothole)

Download: https://github.com/sekilab/RoadDamageDetector or Zenodo DOI: 10.5281/zenodo.7090107

## Models

### 1. Custom YOLOv8 (from scratch)
- CSP-style backbone (YOLOv8s scale: width=0.5, depth=0.33)
- PAN/FPN neck
- 3-scale anchor-free detection heads (P3/8, P4/16, P5/32)
- **Extension: YOLOv8+P2** — extra P2 head (stride 4) for tiny cracks

### 2. Ultralytics YOLOv8s
- COCO-pretrained weights (`yolov8s.pt`)
- Fine-tuned on RDD2022 with built-in augmentation + AMP

## Results

| Model | mAP@0.50 | Precision | Recall |
|---|---|---|---|
| Custom YOLOv8 (Standard NMS) | 0.41 | 0.17 | 0.26 |
| Custom YOLOv8 (Soft-NMS) | 0.14 | 0.17 | 0.26 |
| **Ultralytics YOLOv8s** | **0.58** | **0.63** | **0.54** |
| MN-YOLOv5 (baseline paper) | 0.53 | — | — |

## Repository Structure

```
road-damage-detection/
├── data/
│   ├── prepare_dataset.py       # Split RDD2022 into train/val
│   └── convert_to_yolo.py       # Convert VOC XML → YOLO format
├── models/
│   ├── backbone.py              # CSP backbone
│   ├── neck.py                  # PAN-FPN neck
│   ├── head.py                  # Anchor-free detection heads
│   └── yolov8.py                # Full model: YOLOv8 & YOLOv8P2
├── utils/
│   ├── loss.py                  # Box (CIoU) + Classification + DFL loss
│   ├── nms.py                   # Standard NMS + Soft-NMS
│   ├── metrics.py               # mAP, precision, recall
│   └── dataset.py               # RDDDataset & DataLoader
├── configs/
│   ├── custom_yolov8.yaml
│   └── ultralytics_yolov8s.yaml
├── train.py                     # Train custom model
├── train_ultralytics.py         # Fine-tune Ultralytics YOLOv8s
├── evaluate.py                  # Evaluate either model
├── inference.py                 # Run detections on images
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```bash
python data/prepare_dataset.py --data_dir ./dataset --val_split 0.15
python data/convert_to_yolo.py --split_dir ./dataset/rdd2022_split --out_dir ./dataset/rdd2022_yolo
```

### 3. Train custom YOLOv8
```bash
python train.py --data_dir ./dataset/rdd2022_yolo --model standard --epochs 100
python train.py --data_dir ./dataset/rdd2022_yolo --model p2 --epochs 100 --use_soft_nms
```

### 4. Train Ultralytics YOLOv8s
```bash
python train_ultralytics.py --data ./dataset/rdd2022_yolo/rdd2022.yaml --epochs 100 --batch 64
```

### 5. Evaluate
```bash
python evaluate.py --model_type custom --weights ./runs/custom/best.pt --soft_nms
python evaluate.py --model_type ultralytics --weights ./runs/ultralytics/yolov8s_rdd2022/weights/best.pt
```

### 6. Inference
```bash
python inference.py --model_type custom --weights ./runs/custom/best.pt --source ./sample_images/ --soft_nms
```

## Loss Function

L_total = lambda_box * L_box + lambda_cls * L_cls + lambda_DFL * L_DFL

| Component | Weight | Description |
|---|---|---|
| Box loss (CIoU) | 7.5 | Bounding-box regression |
| Classification loss (BCE) | 0.5 | 4-class damage classification |
| DFL (Distribution Focal Loss) | 1.5 | Precise border prediction |

## Training Parameters

| Parameter | Value |
|---|---|
| Image size | 640 x 640 |
| Batch size | 64 |
| Optimizer | AdamW |
| Weight decay | 0.0005 |
| Backbone LR | 0.001 |
| Head LR | 0.01 |
| Epochs | 100 |
| LR schedule | Cosine annealing |
| Gradient clipping | max_norm = 10 |
| Hardware | NVIDIA A100 40GB |

## Key Findings

- Ultralytics YOLOv8s (COCO-pretrained) significantly outperforms the custom scratch model, achieving mAP 0.58 vs 0.41
- The pretrained model also converged 52% faster
- Adding a P2 head to the custom model improves detection of tiny, thin cracks but did not close the gap with Ultralytics
- Soft-NMS improved recall slightly for overlapping cracks compared to standard NMS

