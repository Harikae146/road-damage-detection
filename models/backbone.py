"""
backbone.py
-----------
CSP (Cross Stage Partial) backbone - the feature-extraction spine of our
custom YOLOv8. Architecture mirrors YOLOv8s in depth and width scaling.

Produces feature maps at four strides:
    P2  /4   (stride 4)   <- used by P2 detection head for tiny cracks
    P3  /8   (stride 8)   <- small objects
    P4  /16  (stride 16)  <- medium objects
    P5  /32  (stride 32)  <- large objects
"""

import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    """Standard Conv -> BN -> SiLU block."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv8-style bottleneck with optional residual connection."""
    def __init__(self, ch, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(ch * expansion)
        self.cv1 = ConvBNSiLU(ch, hidden, 3, 1)
        self.cv2 = ConvBNSiLU(hidden, ch, 3, 1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Fuse block (C2f) - the main repeatable unit in YOLOv8 backbone.
    Splits channels, passes half through n bottlenecks, then concatenates.
    """
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.hidden = int(out_ch * expansion)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.hidden, 1, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.hidden, out_ch, 1, 1)
        self.m   = nn.ModuleList(
            Bottleneck(self.hidden, shortcut, expansion=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (max-pool at 3 scales then concat)."""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, hidden, 1, 1)
        self.cv2 = ConvBNSiLU(hidden * 4, out_ch, 1, 1)
        self.mp  = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPBackbone(nn.Module):
    """
    YOLOv8s-scale CSP backbone.
    Channel widths follow the YOLOv8s multiplier (width=0.50, depth=0.33).
    Returns a dict with keys 'P2', 'P3', 'P4', 'P5' for the neck.
    """
    def __init__(self, in_ch=3, width_mult=0.50, depth_mult=0.33):
        super().__init__()

        def ch(c):   return max(1, int(c * width_mult))
        def rep(n):  return max(1, round(n * depth_mult))

        # Stem - stride 2
        self.stem = ConvBNSiLU(in_ch, ch(64), 3, 2)

        # Stage 1 - stride 2 -> P2 (1/4)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(ch(64), ch(128), 3, 2),
            C2f(ch(128), ch(128), n=rep(3), shortcut=True),
        )

        # Stage 2 - stride 2 -> P3 (1/8)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(ch(128), ch(256), 3, 2),
            C2f(ch(256), ch(256), n=rep(6), shortcut=True),
        )

        # Stage 3 - stride 2 -> P4 (1/16)
        self.stage3 = nn.Sequential(
            ConvBNSiLU(ch(256), ch(512), 3, 2),
            C2f(ch(512), ch(512), n=rep(6), shortcut=True),
        )

        # Stage 4 - stride 2 -> P5 (1/32)
        self.stage4 = nn.Sequential(
            ConvBNSiLU(ch(512), ch(512), 3, 2),
            C2f(ch(512), ch(512), n=rep(3), shortcut=True),
            SPPF(ch(512), ch(512), k=5),
        )

    def forward(self, x):
        p1 = self.stem(x)          # /2
        p2 = self.stage1(p1)       # /4
        p3 = self.stage2(p2)       # /8
        p4 = self.stage3(p3)       # /16
        p5 = self.stage4(p4)       # /32
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}
