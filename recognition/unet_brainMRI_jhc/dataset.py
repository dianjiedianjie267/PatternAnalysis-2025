# dataset.py
# OASIS slice dataset loader (2D PNG slices + masks)

import os
import re
import glob
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset

# 容错：有些 PNG 可能是截断的
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_valid_png(path: str) -> bool:
    """用 PIL.verify() 检查 PNG 是否真的能打开"""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class OasisSliceDataset(Dataset):
    """
    root_dir 结构示例:
        root_dir/
          keras_png_slices_train/
          keras_png_slices_validate/
          keras_png_slices_test/
          keras_png_slices_seg_train/
          keras_png_slices_seg_validate/
          keras_png_slices_seg_test/

    图像示例: case_401_slice_26.nii.png
    掩码示例: seg_401_slice_26.nii.png
    """

    def __init__(self, root_dir, split="train", transform=None):
        # 允许 val/validate
        if split == "val":
            split = "validate"
        assert split in ["train", "validate", "test"], \
            "split must be train/val(validate)/test"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        img_dir = os.path.join(root_dir, f"keras_png_slices_{split}")
        seg_dir = os.path.join(root_dir, f"keras_png_slices_seg_{split}")

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Image dir not found: {img_dir}")
        if not os.path.isdir(seg_dir):
            raise RuntimeError(f"Mask dir not found:  {seg_dir}")

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        index = []

        for img_path in img_paths:
            base = os.path.basename(img_path)  # e.g. case_401_slice_26.nii.png

            # 规则1：case_XXXX -> seg_XXXX
            mask_base = re.sub(r"^case_", "seg_", base)
            mask_path = os.path.join(seg_dir, mask_base)

            # 兜底：也许是 foo.png -> foo_seg.png
            if not os.path.exists(mask_path):
                if base.lower().endswith(".png"):
                    stem = base[:-4]
                else:
                    stem = base
                alt = stem + "_seg.png"
                mask_path_alt = os.path.join(seg_dir, alt)
                if os.path.exists(mask_path_alt):
                    mask_path = mask_path_alt

            if not os.path.exists(mask_path):
                # 没有对应mask就跳过
                print(f"[WARN] No mask for {img_path} -> expected {mask_base}, skipping")
                continue

            # 在索引阶段验证一次PNG有效性
            if not is_valid_png(img_path):
                print(f"[WARN] Bad/corrupted image, skipping: {img_path}")
                continue
            if not is_valid_png(mask_path):
                print(f"[WARN] Bad/corrupted mask, skipping:  {mask_path}")
                continue

            index.append((img_path, mask_path))

        self.index = index
        print(f"[INFO] {split} split: {len(self.index)} slices indexed "
              f"from {img_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        img_path, mask_path = self.index[i]

        # 打开图像和掩码
        try:
            img_pil = Image.open(img_path).convert("L")  # 灰度
            msk_pil = Image.open(mask_path).convert("L") # 灰度 (label map)
        except Exception as e:
            # 如果偶尔读失败（极少）可以抛 IndexError 让 DataLoader 重试
            raise IndexError(f"Failed to open: {img_path} / {mask_path}: {e}")

        img_np = np.array(img_pil).astype(np.float32)  # (H,W)
        msk_np = np.array(msk_pil).astype(np.int64)    # (H,W)  可能是0/255

        # 归一化图像强度到0~1
        vmax = float(img_np.max())
        if vmax > 0:
            img_np /= vmax

        img_t = torch.from_numpy(img_np).unsqueeze(0).float()  # (1,H,W)
        msk_t = torch.from_numpy(msk_np).long()                 # (H,W)

        sample = {
            "image": img_t,
            "mask": msk_t,
            "image_path": img_path,
            "mask_path": mask_path,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
