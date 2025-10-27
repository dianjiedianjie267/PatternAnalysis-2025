# train.py
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import OasisSliceDataset
from modules import UNet


def train_one_epoch(model, loader, optimizer, device):
    """
    单个epoch的训练循环:
    - 前向
    - BCEWithLogitsLoss
    - 反向传播
    - 优化
    - 返回平均loss和耗时
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    t0 = time.time()

    for batch in loader:
        imgs = batch["image"].to(device)      # (B,1,H,W) float32 [0,1]
        masks = batch["mask"].to(device)      # (B,H,W) int64, 通常是0或255

        # 把mask二值化成0/1，避免出现255导致loss爆炸
        masks_bin = (masks > 0).float()       # (B,H,W) float32
        masks_f   = masks_bin.unsqueeze(1)    # (B,1,H,W)

        optimizer.zero_grad()

        logits = model(imgs)                  # (B,1,H,W) raw logits
        loss = F.binary_cross_entropy_with_logits(logits, masks_f)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    t1 = time.time()
    return avg_loss, (t1 - t0)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    """
    验证集评估:
    - val_loss (同样的BCEWithLogitsLoss)
    - val_dice 计算 (0.5阈值二值化)
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    for batch in loader:
        imgs = batch["image"].to(device)      # (B,1,H,W)
        masks = batch["mask"].to(device)      # (B,H,W)

        # 二值化mask
        masks_bin = (masks > 0).float()       # (B,H,W)
        masks_f   = masks_bin.unsqueeze(1)    # (B,1,H,W)

        logits = model(imgs)                  # (B,1,H,W)
        loss = F.binary_cross_entropy_with_logits(logits, masks_f)

        # Dice:
        # 1) 概率化
        probs = torch.sigmoid(logits)         # (B,1,H,W)
        # 2) 二值化预测
        preds = (probs > 0.5).float()         # (B,1,H,W)

        # 3) dice = 2|pred ∩ gt| / (|pred| + |gt|)
        intersection = (preds * masks_f).sum(dim=[1,2,3])
        union = preds.sum(dim=[1,2,3]) + masks_f.sum(dim=[1,2,3])
        dice_batch = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_mean = dice_batch.mean()

        running_loss += loss.item()
        running_dice += dice_mean.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_dice = running_dice / max(n_batches, 1)
    return avg_loss, avg_dice


def main():
    # -------------------------
    # CLI args
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to OASIS root dir containing keras_png_slices_* folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit_train", type=int, default=0,
                        help="0 = use all; otherwise use first N train slices")
    parser.add_argument("--limit_val", type=int, default=0,
                        help="0 = use all; otherwise use first N val slices")
    args = parser.parse_args()

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = OasisSliceDataset(args.data_root, split="train")
    val_ds   = OasisSliceDataset(args.data_root, split="validate")

    full_train_len = len(train_ds)
    full_val_len   = len(val_ds)

    # 可选子采样（调试/加速用）
    if args.limit_train > 0:
        final_train_len = min(args.limit_train, full_train_len)
        train_ds.index = train_ds.index[:final_train_len]
    else:
        final_train_len = full_train_len

    if args.limit_val > 0:
        final_val_len = min(args.limit_val, full_val_len)
        val_ds.index = val_ds.index[:final_val_len]
    else:
        final_val_len = full_val_len

    # 状态输出
    print(f"[INFO] full train size: {full_train_len}")
    print(f"[INFO] full val   size: {full_val_len}")
    print(f"[INFO] final train size: {final_train_len}")
    print(f"[INFO] final val size:   {final_val_len}")

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,    # 你可以改成2/4加速，但Mac上有时num_workers>0会有问题
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # -------------------------
    # Model & Optimizer
    # -------------------------
    model = UNet(in_channels=1, out_channels=1)  # binary segmentation
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------
    # Training loop (multi-epoch)
    # -------------------------
    best_dice = -1.0
    train_curve = []  # train_loss per epoch
    val_curve = []    # val_loss per epoch
    dice_curve = []   # val_dice per epoch

    for epoch in range(args.epochs):
        train_loss, train_time = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = eval_one_epoch(model, val_loader, device)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_dice={val_dice:.4f} "
            f"time={train_time:.1f}s"
        )

        train_curve.append(train_loss)
        val_curve.append(val_loss)
        dice_curve.append(val_dice)

        # Save best model by dice score
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"[SAVE] best_model.pth updated (best_dice={best_dice:.4f})")

    print(f"[DONE] best_dice={best_dice:.4f}")

    # -------------------------
    # Save training curve plot
    # -------------------------
    try:
        plt.figure(figsize=(6,4))
        plt.plot(train_curve, label="train_loss")
        plt.plot(val_curve, label="val_loss")
        plt.plot(dice_curve, label="val_dice")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=150)
        print("[SAVE] training_curve.png written")
    except Exception as e:
        print("[WARN] could not save training curve:", e)


if __name__ == "__main__":
    main()
