"""
U-Net 分割评价脚本

功能:
1. 载入训练好的 U-Net 权重
2. 在指定数据集上评估分割效果
3. 输出: Acc / Dice / mIoU + Params / FLOPs
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.unet import count_params_flops
from config import Config
from data.dataset import ISICDataset
from model.factory import build_model

# ==============================================================================
# 工具函数
# ==============================================================================
def resolve_model_path(cfg):
    if cfg.MODEL_PATH:
        return cfg.MODEL_PATH
    best = os.path.join(cfg.RUN_DIR, "best.pth")
    last = os.path.join(cfg.RUN_DIR, "last.pth")
    for p in (best, last):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("未找到权重文件，请先训练或在 Config.MODEL_PATH 中指定模型路径。")


def compute_batch_confusion(pred, target):
    """B×H×W 二值掩码 → 逐样本 TP/FP/TN/FN"""
    pred   = pred.float()
    target = target.float()
    tp     = (pred * target).flatten(1).sum(dim=1)
    fp     = pred.flatten(1).sum(dim=1) - tp
    fn     = target.flatten(1).sum(dim=1) - tp
    return tp, fp, fn


# ==============================================================================
# 评估
# ==============================================================================
@torch.no_grad()
def evaluate(model, loader, cfg):
    accum = torch.zeros(3, device=cfg.DEVICE)   # Acc, Dice, mIoU
    total_samples = 0

    for images, masks in (pbar := tqdm(loader, desc="Eval", ncols=100)):
        images = images.to(cfg.DEVICE, non_blocking=cfg.PIN_MEMORY)
        masks  = masks.to(cfg.DEVICE, non_blocking=cfg.PIN_MEMORY)

        pred_bin = (torch.sigmoid(model(images)) > cfg.THRESHOLD).float()
        tp, fp, fn = compute_batch_confusion(pred_bin, masks)

        # 逐样本指标
        tn     = (pred_bin == masks).float().flatten(1).sum(dim=1) - tp
        acc    = (tp + tn) / (tp + fp + tn + fn + 1e-6)
        dice   = (2. * tp + 1e-6) / (tp + fp + tp + fn + 1e-6)
        iou    = (tp + 1e-6) / (tp + fp + fn + 1e-6)   # 单类 mIoU ≡ IoU

        accum += torch.stack([acc.sum(), dice.sum(), iou.sum()])
        total_samples += images.size(0)

        pbar.set_postfix(dice=f"{dice.mean().item():.4f}", iou=f"{iou.mean().item():.4f}")

    return (accum / total_samples).cpu().numpy()


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    cfg = Config(mode="eval")

    # ---- 数据加载 ----
    dataset = ISICDataset(data_dir=cfg.DATA_DIR, split=cfg.SPLIT, img_size=cfg.IMG_SIZE)
    if isinstance(dataset[0][1], str):
        raise RuntimeError(f"当前 {cfg.SPLIT} 集无 GT mask，无法评估。")
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    print(f"评价数据集: {cfg.SPLIT}")
    print(f"样本数量  : {len(dataset)}")

    # ---- 加载模型 ----
    model_path = resolve_model_path(cfg)
    model = build_model(cfg)
    ckpt = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"加载模型  : {model_path}")
    print(f"来源 epoch: {ckpt.get('epoch', '?')}  |  历史最佳 Dice: {ckpt.get('best_dice', 0):.4f}")

    # ---- 模型复杂度 ----
    params_m, flops_g = count_params_flops(model, input_shape=(3, cfg.IMG_SIZE, cfg.IMG_SIZE),
                                            device=cfg.DEVICE)
    print(f"参数量    : {params_m:.2f} M")
    print(f"FLOPs     : {flops_g:.2f} GFLOPs")

    # ---- 评估 ----
    means = evaluate(model, loader, cfg)

    print("\n================ 评价结果 ================")
    print(f"Acc     : {means[0]:.4f}")
    print(f"Dice    : {means[1]:.4f}")
    print(f"mIoU    : {means[2]:.4f}")
    print(f"Params  : {params_m:.2f} M")
    print(f"FLOPs   : {flops_g:.2f} GFLOPs")
    print("==========================================")


if __name__ == "__main__":
    main()
