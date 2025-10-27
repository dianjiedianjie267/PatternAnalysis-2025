"""
modules.py
Defines:
- UNet (2D U-Net style segmentation network)
- dice_coeff() for evaluation
- dice_loss() for training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(pred_logits, target_mask, epsilon=1e-6):
    """
    pred_logits: (B, 1, H, W) raw logits from model
    target_mask: (B, H, W) or (B,1,H,W) integer mask
                 can be 0/1 or 0/255 etc.

    We'll binarize target_mask > 0 and also binarize prediction at 0.5.
    Returns mean dice across batch.
    """
    # ensure shapes
    if target_mask.dim() == 3:
        # (B,H,W) -> (B,1,H,W)
        target_mask = target_mask.unsqueeze(1)

    # binarize gt (0/1 float)
    target_bin = (target_mask > 0).float()  # (B,1,H,W)

    # prediction -> probability -> binary
    probs = torch.sigmoid(pred_logits)      # (B,1,H,W)
    preds = (probs > 0.5).float()           # (B,1,H,W)

    # dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
    intersection = (preds * target_bin).sum(dim=[1, 2, 3])
    union = preds.sum(dim=[1, 2, 3]) + target_bin.sum(dim=[1, 2, 3])
    dice_per_item = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice_per_item.mean()


def dice_loss(pred_logits, target_mask):
    """
    1 - dice_coeff
    """
    return 1.0 - dice_coeff(pred_logits, target_mask)


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


class UNet(nn.Module):
    """
    Minimal 2D U-Net style encoder-decoder for binary segmentation.

    in_channels:  number of channels in input image (1 for grayscale MRI slice)
    out_channels: number of channels in output mask (1 for binary mask)
    base_ch:      width of first conv layer
    """

    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = DoubleConv(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4,
                                      kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch,
                                      kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Output head
        self.out_head = nn.Conv2d(base_ch, out_channels, kernel_size=1)

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

        out = self.out_head(c1)  # (B, out_channels, H, W) raw logits
        return out
