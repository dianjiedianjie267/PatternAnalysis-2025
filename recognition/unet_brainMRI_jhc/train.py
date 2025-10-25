"""
train.py
Train the UNet2D model on OasisSliceDataset.

Usage example (on cluster / HPC):
    python train.py \
        --data_root /home/groups/comp3710/OASIS \
        --epochs 20 \
        --batch_size 4 \
        --lr 1e-3

What this script does:
- build train/val dataloaders
- loop epochs: forward/backward/step
- calculate validation Dice per epoch
- save best checkpoint to best_model.pth
- save training curve plot training_curve.png
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import OasisSliceDataset
from modules import UNet2D, dice_loss, dice_coeff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2D U-Net on OASIS brain MRI slices"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the OASIS dataset root directory (with images + masks)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam."
    )
    return parser.parse_args()


def train_model(data_root, num_epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Datasets
    train_ds = OasisSliceDataset(root_dir=data_root, split="train")
    val_ds   = OasisSliceDataset(root_dir=data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = UNet2D(n_channels=1, n_classes=1).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    train_loss_hist = []
    val_dice_hist = []

    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)   # (B,1,H,W)
            masks = batch["mask"].to(device)   # (B,H,W)

            logits = model(imgs)               # (B,1,H,W)
            loss = dice_loss(logits, masks)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        train_loss_hist.append(avg_loss)

        # ----- validation -----
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                d = dice_coeff(logits, masks)
                dices.append(d.item())

        avg_dice = sum(dices) / max(1, len(dices))
        val_dice_hist.append(avg_dice)

        print(f"[EPOCH {epoch+1}/{num_epochs}] "
              f"loss={avg_loss:.4f}  val_dice={avg_dice:.4f}")

        # save best checkpoint
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(">> [CHECKPOINT] saved best_model.pth")

    # ---- plot curves (to include in README) ----
    plt.figure()
    plt.plot(train_loss_hist, label="train dice loss")
    plt.plot(val_dice_hist, label="val dice coeff")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.title("Training progress")
    plt.savefig("training_curve.png", dpi=200)
    print(">> saved training_curve.png")

    print(f"[DONE] best val Dice = {best_val_dice:.4f}")
    return best_val_dice


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_root=args.data_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
