"""
ISIC 2018 皮肤病变分割数据集

数据集结构:
  dataset/
    ISIC2018_Task1-2_Training_Input/       ← 训练原图
    ISIC2018_Task1_Training_GroundTruth/   ← 训练 mask
    ISIC2018_Task1-2_Validation_Input/     ← 验证原图
    ISIC2018_Task1_Validation_GroundTruth/ ← 验证 mask
    ISIC2018_Task1-2_Test_Input/           ← 测试原图
    ISIC2018_Task1_Test_GroundTruth/       ← 测试 mask

数据增强使用 albumentations，确保几何变换同步作用于 image 和 mask。
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================================================================
# 数据增强配置
# ==============================================================================
def get_train_transforms(img_size=256):
    """训练阶段增强：几何 + 颜色 + 噪声，同步作用于 image 和 mask"""
    return A.Compose([
        # --- 几何增强 (mask 同步变换) ---
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0),
                            ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),

        # --- 颜色增强 (只作用于 image) ---
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.6),

        # --- 噪声 / 模糊 (模拟皮肤镜图像质量差异) ---
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),

        # --- 弹性变形 (模拟皮肤拉伸) ---
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # --- 归一化 ---
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    """验证阶段：只做 resize + 归一化"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# 原图可视化用的逆归一化
INV_NORMALIZE = dict(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


# ==============================================================================
# ISIC 数据集
# ==============================================================================
class ISICDataset(Dataset):
    """
    ISIC 2018 皮肤病变分割数据集

    Args:
        data_dir   : 数据集根目录 (dataset/)
        split      : 'train' / 'val' / 'test'
        transform  : albumentations Compose (如果为 None 则只 resize + 归一化)
        img_size   : resize 目标尺寸
    """

    def __init__(self, data_dir, split='train', transform=None, img_size=256):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size

        if split == 'train':
            self.img_dir = os.path.join(data_dir, "ISIC2018_Task1-2_Training_Input")
            self.mask_dir = os.path.join(data_dir, "ISIC2018_Task1_Training_GroundTruth")
        elif split == 'val':
            self.img_dir = os.path.join(data_dir, "ISIC2018_Task1-2_Validation_Input")
            self.mask_dir = os.path.join(data_dir, "ISIC2018_Task1_Validation_GroundTruth")
        elif split == 'test':
            self.img_dir = os.path.join(data_dir, "ISIC2018_Task1-2_Test_Input")
            test_mask_dir = os.path.join(data_dir, "ISIC2018_Task1_Test_GroundTruth")
            self.mask_dir = test_mask_dir if os.path.exists(test_mask_dir) else None
        else:
            raise ValueError(f"split must be 'train'/'val'/'test', got {split}")

        self.ids = sorted(
            [f.replace('.jpg', '') for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        )

        if transform is None:
            self.transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 读取原图 (BGR → RGB)
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.split == 'test' and self.mask_dir is None:
            # 测试集无标注时，仅返回图像与 ID
            transformed = self.transform(image=img)
            return transformed['image'], img_id

        # 读取 mask (灰度图)
        mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # → {0, 1}

        # albumentations 同步变换
        transformed = self.transform(image=img, mask=mask)

        return transformed['image'], transformed['mask'].unsqueeze(0)  # (1, H, W)


def get_dataloaders(data_dir, img_size=256, batch_size=8, num_workers=0):
    """
    快速获取 train / val / test 三个 DataLoader

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = ISICDataset(data_dir, split='train', img_size=img_size)
    val_ds = ISICDataset(data_dir, split='val', img_size=img_size)
    test_ds = ISICDataset(data_dir, split='test', img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ==============================================================================
# 测试入口
# ==============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DATA_DIR = r"d:\code\segment\dataset"

    # 测试 train / val / test 三个 split
    for split_name, split in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
        ds = ISICDataset(DATA_DIR, split=split, img_size=256)
        print(f"\n=== {split_name} ===")
        print(f"  样本数: {len(ds)}")
        sample = ds[0]
        if isinstance(sample[1], str):
            img_tensor, img_id = sample
            print(f"  image shape: {tuple(img_tensor.shape)}  ID: {img_id}")
        else:
            img_tensor, mask_tensor = sample
            print(f"  image shape : {tuple(img_tensor.shape)}  dtype: {img_tensor.dtype}")
            print(f"  mask  shape : {tuple(mask_tensor.shape)}  dtype: {mask_tensor.dtype}")
            print(f"  mask  unique: {mask_tensor.unique().tolist()}")
            print(f"  image range : [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

    # Loader
    tl, vl, _ = get_dataloaders(DATA_DIR, img_size=256, batch_size=4)
    imgs, masks = next(iter(tl))
    print(f"\n=== DataLoader ===")
    print(f"  Train batch: {tuple(imgs.shape)}, mask: {tuple(masks.shape)}")
    imgs_v, masks_v = next(iter(vl))
    print(f"  Val   batch: {tuple(imgs_v.shape)}, mask: {tuple(masks_v.shape)}")
