"""
prepare_dataset.py
------------------
Download and organise the RDD2022 dataset, then split into train/val sets.

Usage:
    python data/prepare_dataset.py --data_dir ./dataset --val_split 0.15

The script expects the RDD2022 zip archives to be placed in `data_dir` first.
Download from: https://github.com/sekilab/RoadDamageDetector
or Zenodo (DOI: 10.5281/zenodo.7090107)
"""

import os
import random
import shutil
import argparse
from pathlib import Path


COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike", "China_Drone"]
DAMAGE_CLASSES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def find_image_label_pairs(root: Path):
    """Walk root and return matched (image_path, annotation_path) pairs."""
    pairs = []
    for img_path in sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png")):
        label_path = (
            img_path.parent.parent / "annotations" / "xmls" / (img_path.stem + ".xml")
        )
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def split_pairs(pairs, val_split=0.15, seed=42):
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    return shuffled[n_val:], shuffled[:n_val]


def copy_split(pairs, dest_images: Path, dest_labels: Path):
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, dest_images / img_path.name)
        shutil.copy2(lbl_path, dest_labels / lbl_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--out_dir", default="./dataset/rdd2022_split")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_dir)
    out = Path(args.out_dir)

    all_pairs = find_image_label_pairs(root)
    print(f"Found {len(all_pairs)} image-annotation pairs.")

    train_pairs, val_pairs = split_pairs(all_pairs, args.val_split, args.seed)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    copy_split(train_pairs, out / "train" / "images", out / "train" / "labels_xml")
    copy_split(val_pairs,   out / "val"   / "images", out / "val"   / "labels_xml")
    print(f"Dataset split written to {out}")


if __name__ == "__main__":
    main()
