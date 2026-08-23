"""
head.py
-------
Anchor-free detection heads for the custom YOLOv8.

DetectHead   - single-scale head that outputs raw box + class logits.
MultiScaleHead - wraps N DetectHead instances, one per feature level.

Box encoding follows YOLOv8: each cell predicts 4 * reg_max values
(left, top, right, bottom distribution) decoded with DFL, plus
num_classes BCE-logits.
"""

import math
import torch
import torch.nn as nn
from .backbone import ConvBNSiLU


class DetectHead(nn.Module):
    """
    Single-scale anchor-free detection head.
    Separate branches for box regression and class prediction.
    """
    def __init__(self, in_ch: int, num_classes: int = 4, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        hidden = max(in_ch, 256)

        # Box branch: 2 x ConvBNSiLU -> conv -> 4*reg_max channels
        self.box_branch = nn.Sequential(
            ConvBNSiLU(in_ch, hidden, 3, 1),
            ConvBNSiLU(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 4 * reg_max, 1),
        )

        # Class branch: 2 x ConvBNSiLU -> conv -> num_classes channels
        self.cls_branch = nn.Sequential(
            ConvBNSiLU(in_ch, hidden, 3, 1),
            ConvBNSiLU(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialise bias of the last conv in cls_branch so
        # sigmoid output starts near prior = 0.01  =>  logit ~ -4.6
        nn.init.constant_(self.cls_branch[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            box_pred: (B, 4*reg_max, H, W)  raw DFL outputs
            cls_pred: (B, num_classes, H, W)  raw class logits
        """
        return self.box_branch(x), self.cls_branch(x)


class MultiScaleHead(nn.Module):
    """
    Collection of DetectHead modules, one per FPN level.
    Wraps them and returns a flat list of (box, cls) per scale.
    """
    def __init__(self, in_channels: list, num_classes: int = 4, reg_max: int = 16):
        """
        Args:
            in_channels: list of channel counts, one per scale
                         e.g. [128, 256, 512] for 3-scale
                         or   [64, 128, 256, 512] for 4-scale P2 model
            num_classes: number of damage classes (default 4)
            reg_max: DFL bins (default 16, matching YOLOv8)
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.heads = nn.ModuleList(
            DetectHead(c, num_classes, reg_max) for c in in_channels
        )

    def forward(self, features: list):
        """
        Args:
            features: list of tensors, one per scale (from neck)
        Returns:
            list of (box_pred, cls_pred) tuples, one per scale
        """
        return [head(feat) for head, feat in zip(self.heads, features)]

    @staticmethod
    def decode_box(box_raw: torch.Tensor, reg_max: int, stride: float,
                   device: torch.device) -> torch.Tensor:
        """
        Decode raw DFL box predictions to (x1,y1,x2,y2) in image pixels.
        Args:
            box_raw: (B, 4*reg_max, H, W)
            reg_max: number of DFL bins
            stride:  feature-map stride in pixels
            device:  target device
        Returns:
            boxes: (B, H*W, 4)  in image-pixel coordinates (x1,y1,x2,y2)
        """
        B, _, H, W = box_raw.shape
        # Project each of the 4 distributions
        box = box_raw.view(B, 4, reg_max, H, W)
        box = box.softmax(dim=2)
        proj = torch.arange(reg_max, dtype=torch.float32, device=device)
        # Weighted sum -> (B, 4, H, W)
        box = (box * proj.view(1, 1, reg_max, 1, 1)).sum(dim=2)

        # Build anchor grid
        ys = torch.arange(H, dtype=torch.float32, device=device) + 0.5
        xs = torch.arange(W, dtype=torch.float32, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
        cx = grid_x.unsqueeze(0).expand(B, -1, -1)
        cy = grid_y.unsqueeze(0).expand(B, -1, -1)

        # ltrb -> xyxy (in stride units, then scale to pixels)
        l, t, r, b = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = (cx - l) * stride
        y1 = (cy - t) * stride
        x2 = (cx + r) * stride
        y2 = (cy + b) * stride
        return torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
