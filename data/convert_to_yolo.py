"""
convert_to_yolo.py
------------------
Convert RDD2022 Pascal-VOC XML annotations to YOLO format.

Each output .txt has one row per bounding box:
    class_id  x_center_norm  y_center_norm  width_norm  height_norm

Damage class mapping:
    D00 -> 0   (Longitudinal Crack)
    D10 -> 1   (Transverse Crack)
    D20 -> 2   (Alligator Crack)
    D40 -> 3   (Pothole)

Usage:
    python data/convert_to_yolo.py \\
        --split_dir ./dataset/rdd2022_split \\
        --out_dir   ./dataset/rdd2022_yolo
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import shutil


CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append((cls_name, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


def to_yolo_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_split(images_dir, xml_dir, out_images, out_labels):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    converted, skipped = 0, 0
    all_imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    for img_path in all_imgs:
        xml_path = xml_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            skipped += 1
            continue
        img_w, img_h, boxes = parse_xml(xml_path)
        lines = []
        for cls_name, xmin, ymin, xmax, ymax in boxes:
            cls_id = CLASS_MAP[cls_name]
            lines.append(to_yolo_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h))
        shutil.copy2(img_path, out_images / img_path.name)
        label_file = out_labels / (img_path.stem + ".txt")
        label_file.write_text("\n".join(lines))
        converted += 1
    return converted, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="./dataset/rdd2022_split")
    parser.add_argument("--out_dir", default="./dataset/rdd2022_yolo")
    args = parser.parse_args()

    split_root = Path(args.split_dir)
    out_root   = Path(args.out_dir)

    for split in ("train", "val"):
        images_dir = split_root / split / "images"
        xml_dir    = split_root / split / "labels_xml"
        out_images = out_root / split / "images"
        out_labels = out_root / split / "labels"
        n, skip = convert_split(images_dir, xml_dir, out_images, out_labels)
        print(f"[{split}] Converted {n} images ({skip} skipped).")

    yaml_content = f"""# RDD2022 dataset config
path: {out_root.resolve()}
train: train/images
val: val/images

nc: 4
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
"""
    yaml_path = out_root / "rdd2022.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Dataset YAML written to {yaml_path}")


if __name__ == "__main__":
    main()
