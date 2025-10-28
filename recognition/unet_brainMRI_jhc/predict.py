#!/usr/bin/env python3
"""
predict.py
Run inference with a trained UNet on one MRI slice and save the prediction.

Usage:
    python predict.py \
        --data_root /path/to/OASIS \
        --checkpoint best_model.pth \
        --out prediction_example.png
"""

import argparse
import torch
import matplotlib.pyplot as plt

from dataset import OasisSliceDataset
from modules import UNet

@torch.no_grad()
def run_inference(data_root, checkpoint_path, out_path, device):
    # 1. load dataset (use validate split for demo)
    ds = OasisSliceDataset(data_root, split="validate")
    if len(ds) == 0:
        raise RuntimeError("No data found in validate split")

    # just take the first slice for demo
    sample = ds[0]
    img_t = sample["image"].unsqueeze(0).to(device)   # (1,1,H,W)
    mask_t = sample["mask"].to(device)                # (H,W)

    # 2. build model and load weights
    model = UNet(in_channels=1, out_channels=1)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 3. forward pass
    logits = model(img_t)                # (1,1,H,W), raw logits
    probs = torch.sigmoid(logits)        # (1,1,H,W)
    pred_bin = (probs > 0.5).float()     # (1,1,H,W)

    # 4. move to cpu numpy for plotting
    img_np   = img_t[0,0].cpu().numpy()          # (H,W)
    gt_np    = (mask_t > 0).float().cpu().numpy()# (H,W)
    pred_np  = pred_bin[0,0].cpu().numpy()       # (H,W)

    # 5. plot side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].imshow(img_np, cmap="gray")
    axs[0].set_title("Input MRI")
    axs[0].axis("off")

    axs[1].imshow(gt_np, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_np, cmap="gray")
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[SAVE] wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to OASIS root dir (with keras_png_slices_* etc.)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                        help="Path to trained weights file")
    parser.add_argument("--out", type=str, default="prediction_example.png",
                        help="Output image (visualisation)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    run_inference(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        device=device,
    )

if __name__ == "__main__":
    main()
