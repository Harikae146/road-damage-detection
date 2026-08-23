"""
train.py
--------
Full training loop for the custom YOLOv8 on RDD2022.

Key features:
  - AdamW with separate parameter groups for backbone vs. head
  - Cosine annealing LR schedule with linear warm-up
  - Automatic Mixed Precision (AMP) via torch.cuda.amp.GradScaler
  - Gradient clipping (max_norm = 10)
  - Best/last checkpoint saving
  - mAP@0.50 validation after every epoch

Usage:
  python train.py --data_dir ./dataset/rdd2022_yolo \
                  --model standard \
                  --epochs 100 \
                  --batch_size 64
"""

import argparse
import os
import math
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from models import YOLOv8, YOLOv8P2
from utils.dataset import build_dataloader
from utils.loss import YOLOv8Loss
from utils.metrics import compute_map
from utils.nms import nms_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train custom YOLOv8 on RDD2022')
    p.add_argument('--data_dir',   type=str, required=True,
                   help='Path to rdd2022_yolo/ (must contain train/ and val/)')
    p.add_argument('--model',      type=str, default='standard',
                   choices=['standard', 'p2'],
                   help='"standard" = 3-scale, "p2" = 4-scale with P2 head')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--img_size',   type=int, default=640)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--backbone_lr',type=float, default=1e-3)
    p.add_argument('--head_lr',    type=float, default=1e-2)
    p.add_argument('--weight_decay',type=float, default=5e-4)
    p.add_argument('--grad_clip',  type=float, default=10.0)
    p.add_argument('--warmup',     type=int,   default=3,
                   help='Warm-up epochs with linear LR ramp')
    p.add_argument('--save_dir',   type=str,   default='./runs/custom')
    p.add_argument('--use_soft_nms', action='store_true')
    p.add_argument('--conf_thresh',  type=float, default=0.25)
    p.add_argument('--iou_thresh',   type=float, default=0.45)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(epoch: int, total: int, warmup: int) -> float:
    """Return LR multiplier for current epoch."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, conf_thresh, iou_thresh, use_soft_nms):
    model.eval()
    all_preds, all_gts = [], []

    for batch in val_loader:
        images = batch['images'].to(device)
        raw    = model(images)

        from models.head import MultiScaleHead
        B = images.shape[0]
        scale_boxes, scale_scores = [], []

        for scale_idx, (box_raw, cls_raw) in enumerate(raw):
            stride = model.strides[scale_idx]
            boxes  = MultiScaleHead.decode_box(box_raw, model.reg_max,
                                               stride, device)
            scores = cls_raw.flatten(2).permute(0, 2, 1).sigmoid()
            scale_boxes.append(boxes)
            scale_scores.append(scores)

        all_boxes  = torch.cat(scale_boxes,  dim=1)
        all_scores = torch.cat(scale_scores, dim=1)

        for b in range(B):
            s, l = all_scores[b].max(dim=-1)
            mask  = s > conf_thresh
            boxes_b  = all_boxes[b][mask]
            scores_b = s[mask]
            labels_b = l[mask]

            if boxes_b.shape[0]:
                keep = nms_per_class(boxes_b, scores_b, labels_b,
                                     iou_threshold=iou_thresh,
                                     use_soft=use_soft_nms)
                all_preds.append({'boxes':  boxes_b[keep],
                                  'scores': scores_b[keep],
                                  'labels': labels_b[keep]})
            else:
                all_preds.append({'boxes':  boxes_b,
                                  'scores': scores_b,
                                  'labels': labels_b})

            all_gts.append({'boxes':  batch['boxes'][b].to(device),
                            'labels': batch['labels'][b].to(device)})

    metrics = compute_map(all_preds, all_gts, num_classes=4)
    model.train()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Model
    model = (YOLOv8P2 if args.model == 'p2' else YOLOv8)(num_classes=4)
    model = model.to(device)
    print(f'Model: {type(model).__name__}  |  Device: {device}')

    # Data
    train_loader, val_loader = build_dataloader(
        args.data_dir, args.img_size, args.batch_size, args.num_workers)
    print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')

    # Loss
    criterion = YOLOv8Loss(num_classes=4, reg_max=16,
                           strides=model.strides)

    # Optimizer: separate LRs for backbone vs. head
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.neck.parameters()) +
                       list(model.head.parameters()))
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params,     'lr': args.head_lr},
    ], weight_decay=args.weight_decay)

    scaler    = GradScaler()
    best_map  = 0.0

    for epoch in range(args.epochs):
        model.train()
        lr_mult = cosine_lr(epoch, args.epochs, args.warmup)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * lr_mult if 'initial_lr' in pg                        else pg['lr']

        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images  = batch['images'].to(device)
            targets = [{'boxes':  b.to(device), 'labels': l.to(device)}
                       for b, l in zip(batch['boxes'], batch['labels'])]

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, targets, args.img_size)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        metrics  = validate(model, val_loader, device,
                            args.conf_thresh, args.iou_thresh,
                            args.use_soft_nms)
        mAP = metrics['mAP']

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'loss={avg_loss:.4f} | mAP@0.50={mAP:.4f} | '
              f'P={metrics["mean_precision"]:.3f} | '
              f'R={metrics["mean_recall"]:.3f}')

        # Save checkpoints
        ckpt = {'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'mAP': mAP}
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        if mAP > best_map:
            best_map = mAP
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f'  -> New best mAP: {best_map:.4f}')

    print(f'Training complete. Best mAP@0.50: {best_map:.4f}')


if __name__ == '__main__':
    main()
