"""
train.py
Train the UNet2D model on OasisSliceDataset.

- build train/val dataloaders
- loop epochs: forward/backward/step
- calculate validation Dice
- save best checkpoint
- save training curve plot for README
"""

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import OasisSliceDataset
from modules import UNet2D, dice_loss, dice_coeff

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: point to actual dataset path on HPC or local
    train_ds = OasisSliceDataset(root_dir="/path/to/OASIS", split="train")
    val_ds   = OasisSliceDataset(root_dir="/path/to/OASIS", split="val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = UNet2D(n_channels=1, n_classes=1).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5  # can increase later
    train_loss_hist = []
    val_dice_hist = []

    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)  # (B,1,H,W)
            masks = batch["mask"].to(device)  # (B,H,W)

            logits = model(imgs)              # (B,1,H,W)
            loss = dice_loss(logits, masks)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        train_loss_hist.append(avg_loss)

        # validation
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

        print(f"Epoch {epoch+1}/{num_epochs} loss={avg_loss:.4f} val_dice={avg_dice:.4f}")

        # save best model
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(">> saved best_model.pth")

    # plot training curves for README
    plt.figure()
    plt.plot(train_loss_hist, label="train dice loss")
    plt.plot(val_dice_hist, label="val dice coeff")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.title("Training progress")
    plt.savefig("training_curve.png", dpi=200)
    print(">> saved training_curve.png")

if __name__ == "__main__":
    train_model()

