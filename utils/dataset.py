"""
dataset.py
----------
PyTorch Dataset and DataLoader for the RDD2022 road-damage dataset
in YOLO format (converted from Pascal-VOC XML by data/convert_to_yolo.py).

RDDDataset   - loads images + txt labels, applies letterboxing and augmentation
build_dataloader() - convenience wrapper that returns train/val DataLoaders
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# RDD2022 class names (for reference)
CLASS_NAMES = {0: 'D00_Longitudinal_Crack',
               1: 'D10_Transverse_Crack',
               2: 'D20_Alligator_Crack',
               3: 'D40_Pothole'}


def letterbox(img: np.ndarray, target: int = 640, fill: int = 114):
    """
    Resize + pad image to (target x target) without distortion.
    Returns:
        img_lb:  (target, target, 3)  uint8
        ratio:   scale factor applied to both dimensions
        pad_x:   horizontal padding (total, split evenly)
        pad_y:   vertical padding   (total, split evenly)
    """
    h, w = img.shape[:2]
    ratio = min(target / h, target / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = target - new_w
    pad_y = target - new_h
    top,  bottom = pad_y // 2, pad_y - pad_y // 2
    left, right  = pad_x // 2, pad_x - pad_x // 2

    img_lb = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img_lb, ratio, pad_x, pad_y


class RDDDataset(Dataset):
    """
    YOLO-format road-damage dataset.

    Directory structure expected:
        split_dir/
            images/   *.jpg / *.png
            labels/   *.txt   (one file per image, YOLO format)

    Label file format (one row per box):
        <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    """

    def __init__(self, split_dir: str, img_size: int = 640,
                 augment: bool = True):
        self.img_size = img_size
        self.augment  = augment

        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')
        exts    = ('.jpg', '.jpeg', '.png')

        self.samples = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(exts):
                continue
            stem     = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, stem + '.txt')
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, lbl_path: str):
        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(cls)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _augment(self, img: np.ndarray, boxes: np.ndarray):
        """Random horizontal flip + HSV jitter."""
        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if boxes.shape[0]:
                boxes[:, 0] = 1 - boxes[:, 0]   # flip cx

        # HSV jitter
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img_hsv[..., 0] *= 1 + random.uniform(-0.015, 0.015)  # hue
        img_hsv[..., 1] *= 1 + random.uniform(-0.7,   0.7)    # saturation
        img_hsv[..., 2] *= 1 + random.uniform(-0.4,   0.4)    # value
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        img     = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img, boxes

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        boxes, labels = self._load_labels(lbl_path)

        if self.augment and boxes.shape[0] > 0:
            img, boxes = self._augment(img, boxes)

        img_lb, ratio, pad_x, pad_y = letterbox(img, self.img_size)
        img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        # Normalise to [0, 1]
        img_t = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0

        # Convert normalised cx/cy/w/h (relative to original) to
        # pixel x1y1x2y2 in the letterboxed image
        if boxes.shape[0]:
            h_orig, w_orig = img.shape[:2]
            px_cx = boxes[:, 0] * w_orig * ratio + pad_x / 2
            px_cy = boxes[:, 1] * h_orig * ratio + pad_y / 2
            px_w  = boxes[:, 2] * w_orig * ratio
            px_h  = boxes[:, 3] * h_orig * ratio
            x1 = px_cx - px_w / 2
            y1 = px_cy - px_h / 2
            x2 = px_cx + px_w / 2
            y2 = px_cy + px_h / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)

        return {
            'image':  img_t,
            'boxes':  torch.from_numpy(boxes_xyxy).float(),
            'labels': torch.from_numpy(labels).long(),
            'path':   img_path,
        }


def collate_fn(batch):
    """Custom collate that keeps variable-length box lists as a list."""
    images = torch.stack([b['image']  for b in batch])
    boxes  = [b['boxes']  for b in batch]
    labels = [b['labels'] for b in batch]
    paths  = [b['path']   for b in batch]
    return {'images': images, 'boxes': boxes, 'labels': labels, 'paths': paths}


def build_dataloader(data_dir: str,
                     img_size: int = 640,
                     batch_size: int = 16,
                     num_workers: int = 4):
    """
    Build train and val DataLoaders.

    Args:
        data_dir:    root directory with 'train/' and 'val/' sub-folders
        img_size:    input resolution (default 640)
        batch_size:  images per batch
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader
    """
    train_ds = RDDDataset(os.path.join(data_dir, 'train'),
                          img_size=img_size, augment=True)
    val_ds   = RDDDataset(os.path.join(data_dir, 'val'),
                          img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader
