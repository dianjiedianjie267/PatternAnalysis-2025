"""
predict.py
Run inference with a trained UNet2D checkpoint and save a visualisation.

Typical usage on the COMP3710 cluster:

    python predict.py \
        --data_root /home/groups/comp3710/OASIS \
        --idx 0 \
        --ckpt best_model.pth

This script will:
- load a trained checkpoint (best_model.pth)
- grab one slice from the *test* split
- run the model to get a predicted mask
- save prediction_example.png showing:
    (1) input MRI slice
    (2) ground truth mask (if available)
    (3) predicted mask
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import OasisSliceDataset
from modules import UNet2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference demo for brain MRI segmentation"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the dataset root (e.g. /home/groups/comp3710/OASIS)",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Which sample index from the test split to visualise.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best_model.pth",
        help="Checkpoint file to load (state_dict).",
    )
    return parser.parse_args()


def run_inference(data_root, sample_idx, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load test split
    test_ds = OasisSliceDataset(root_dir=data_root, split="test")
    sample = test_ds[sample_idx]

    # Prepare input + GT
    img = sample["image"].unsqueeze(0).to(device)  # (1,1,H,W)
    mask_gt = sample["mask"].cpu().numpy() if "mask" in sample else None

    # Load model
    model = UNet2D(n_channels=1, n_classes=1).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(img)                # (1,1,H,W)
        prob = torch.sigmoid(logits)       # (1,1,H,W)
        pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0, :, :]

    # Visualise
    plt.figure(figsize=(9, 3))

    # panel 1: Input slice
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img.cpu().numpy()[0, 0, :, :], cmap="gray")
    plt.axis("off")

    # panel 2: Ground truth (if available)
    plt.subplot(1, 3, 2)
    if mask_gt is not None:
        plt.title("GT Mask")
        plt.imshow(mask_gt, cmap="gray")
    else:
        plt.title("GT Mask (N/A)")
        plt.imshow(np.zeros_like(pred_mask), cmap="gray")
    plt.axis("off")

    # panel 3: Predicted mask
    plt.subplot(1, 3, 3)
    plt.title("Pred Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_example.png", dpi=200)
    print(">> saved prediction_example.png")


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        data_root=args.data_root,
        sample_idx=args.idx,
        ckpt_path=args.ckpt,
    )
