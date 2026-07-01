"""
U-Net 分割推理脚本

功能:
1. 载入训练好的 U-Net 权重
2. 对单张图像或目录内全部图像进行推理
3. 保存二值 mask 与 mask 叠加图
"""

import os
import cv2
import numpy as np
import torch

from config import Config
from data.dataset import get_val_transforms
from model.factory import build_model


def resolve_model_path(cfg):
    if cfg.MODEL_PATH:
        return cfg.MODEL_PATH

    best_path = os.path.join(cfg.RUN_DIR, "best.pth")
    last_path = os.path.join(cfg.RUN_DIR, "last.pth")

    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    raise FileNotFoundError("未找到权重文件，请先训练或在 Config.MODEL_PATH 中指定模型路径。")


def list_input_files(cfg):
    if cfg.INPUT_MODE == "file":
        return [cfg.INPUT_FILE]

    if cfg.INPUT_MODE != "dir":
        raise ValueError(f"INPUT_MODE 只能是 'dir' 或 'file'，当前为: {cfg.INPUT_MODE}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [
        os.path.join(cfg.INPUT_DIR, name)
        for name in sorted(os.listdir(cfg.INPUT_DIR))
        if name.lower().endswith(exts)
    ]
    return files


def load_model(cfg):
    model_path = resolve_model_path(cfg)
    model = build_model(cfg)

    ckpt = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"加载模型成功: {model_path}")
    if "epoch" in ckpt:
        print(f"来自 epoch: {ckpt['epoch']}")
    if "best_dice" in ckpt:
        print(f"历史最佳 Dice: {ckpt['best_dice']:.4f}")

    return model


def make_overlay(rgb_img, mask_bin):
    overlay = rgb_img.copy()
    mask_region = mask_bin > 0
    overlay[mask_region] = (
        overlay[mask_region] * 0.5 + np.array([255, 80, 80]) * 0.5
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay


@torch.no_grad()
def predict_one(model, image_path, transform, cfg):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    tensor = transform(image=rgb)["image"].unsqueeze(0).to(cfg.DEVICE)
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_bin = (probs > cfg.THRESHOLD).astype(np.uint8) * 255

    overlay = make_overlay(rgb_resized, mask_bin)
    return rgb_resized, probs, mask_bin, overlay


def save_result(image_path, rgb_img, prob_map, mask_bin, overlay, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{stem}_mask.png")
    prob_path = os.path.join(output_dir, f"{stem}_prob.png")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")

    cv2.imwrite(mask_path, mask_bin)
    cv2.imwrite(prob_path, (prob_map * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"已保存: {stem}")
    print(f"  mask   -> {mask_path}")
    print(f"  prob   -> {prob_path}")
    print(f"  overlay-> {overlay_path}")


def main():
    cfg = Config(mode="predict")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = load_model(cfg)
    transform = get_val_transforms(cfg.IMG_SIZE)
    image_files = list_input_files(cfg)

    print(f"待推理图像数量: {len(image_files)}")
    for image_path in image_files:
        rgb_img, prob_map, mask_bin, overlay = predict_one(model, image_path, transform, cfg)
        save_result(image_path, rgb_img, prob_map, mask_bin, overlay, cfg.OUTPUT_DIR)

    print("\n推理完成。")


if __name__ == "__main__":
    main()
