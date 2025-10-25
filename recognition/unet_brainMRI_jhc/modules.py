"""
modules.py
Defines the segmentation model (U-Net style) and loss/metrics.

We implement:
- Simple UNet2D (encoder-decoder with skip connections)
- dice_coeff() for evaluation
- dice_loss() for training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(pred, target, epsilon=1e-6):
    """
    pred: (B, C, H, W) raw logits or probabilities
    target: (B, H, W) integer class labels

    Binary Dice version:
    - Apply sigmoid to pred
    - Compare with target > 0
    """
    pred_prob = torch.sigmoid(pred)
    pred_flat = pred_prob.view(pred_prob.size(0), -1)
    target_bin = (target > 0).float()
    target_flat = target_bin.view(target_bin.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def dice_loss(pred, target):
    return 1.0 - dice_coeff(pred, target)

class DoubleConv(nn.Module):
    """
    Conv -> ReLU -> Conv -> ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    """
    Minimal U-Net style encoder-decoder for 2D segmentation (Easy baseline).
    """

    def __init__(self, n_channels=1, n_classes=1, base_ch=32):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(n_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = DoubleConv(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Output head
        self.out_head = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        mid = self.mid(p3)

        # Decoder + skip connections
        u3 = self.up3(mid)
        c3 = self.dec3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))

        out = self.out_head(c1)
        return out
