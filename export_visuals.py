"""
导出测试集可视化结果

功能:
读取 test 集，随机抽取 N 个样本，生成:
[原图 | GT 叠加图 | 预测叠加图] 的拼接对比图，供挑选用于论文或 PPT。
"""

import os
import cv2
import numpy as np
import torch
import random
from tqdm import tqdm

from config import Config
from data.dataset import ISICDataset
from model.factory import build_model


def resolve_model_path(cfg):
    if cfg.MODEL_PATH: return cfg.MODEL_PATH
    best = os.path.join(cfg.RUN_DIR, "best.pth")
    last = os.path.join(cfg.RUN_DIR, "last.pth")
    return best if os.path.exists(best) else last


def make_overlay(rgb_img, mask_bin, color=(255, 80, 80), contour_color=(255, 255, 0)):
    """在 RGB 原图上绘制半透明 mask 和边缘轮廓"""
    overlay = rgb_img.copy()
    mask_region = mask_bin > 0
    # 红色半透明叠加
    overlay[mask_region] = (overlay[mask_region] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    # 黄色轮廓
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, contour_color, 2)
    return overlay


def denormalize(tensor):
    """把归一化后的 tensor 转回 0-255 RGB numpy 数组"""
    # 之前这里均值和标准差的符号和顺序有问题，正确逆归一化公式：
    # img = (tensor * std) + mean
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = tensor.cpu().numpy()
    tensor = (tensor * std) + mean
    tensor = np.clip(tensor, 0, 1) * 255.0
    return tensor.transpose(1, 2, 0).astype(np.uint8)


def main():
    cfg = Config(mode="visualize")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 1. 准备模型
    model_path = resolve_model_path(cfg)
    model = build_model(cfg)
    ckpt = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"加载模型成功: {model_path}")

    # 2. 准备数据
    dataset = ISICDataset(data_dir=cfg.DATA_DIR, split=cfg.SPLIT, img_size=cfg.IMG_SIZE)
    if isinstance(dataset[0][1], str):
        raise RuntimeError("当前测试集没有 GT 标注，无法生成对比图。")

    # 随机挑选 N 个样本
    indices = list(range(len(dataset)))
    random.seed(42)  # 固定种子保证每次挑选结果一致
    random.shuffle(indices)
    indices = indices[:cfg.NUM_SAMPLES]

    print(f"开始生成可视化结果至: {cfg.OUTPUT_DIR}")
    for idx in tqdm(indices, desc="Generating"):
        # 由于 dataset[idx] 不返回 img_id，我们从 dataset.ids 中获取
        img_id = dataset.ids[idx]
        image_tensor, mask_tensor = dataset[idx]

        # 模型推理
        with torch.no_grad():
            img_in = image_tensor.unsqueeze(0).to(cfg.DEVICE)
            logits = model(img_in)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # 处理原图和 Mask
        rgb_img = denormalize(image_tensor)
        gt_mask = mask_tensor[0].numpy() > 0.5
        pred_mask = probs > cfg.THRESHOLD

        # 制作叠加图
        # GT 用绿色/蓝色叠加表示？统一用绿色半透明，或者和预测一致用红色
        # 我们用：GT为绿色叠加，预测为红色叠加，以示区分
        gt_overlay = make_overlay(rgb_img, gt_mask, color=(80, 255, 80), contour_color=(0, 255, 0))
        pred_overlay = make_overlay(rgb_img, pred_mask, color=(255, 80, 80), contour_color=(255, 255, 0))

        # 添加标题
        def add_title(img, text):
            # 增加顶部白边
            img_with_border = cv2.copyMakeBorder(img, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # 添加居中黑色文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            cv2.putText(img_with_border, text, (text_x, 22), font, 0.7, (0, 0, 0), 2)
            return img_with_border

        orig_titled = add_title(rgb_img, "Original Image")
        gt_titled = add_title(gt_overlay, "Ground Truth")
        pred_titled = add_title(pred_overlay, "Prediction")

        # 拼接图片： [原图 | GT Overlay | Pred Overlay]
        concat_img = np.concatenate([orig_titled, gt_titled, pred_titled], axis=1)

        # 加点文字（可选，但简单粗暴直接拼接即可）
        # BGR 格式保存
        concat_bgr = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(cfg.OUTPUT_DIR, f"{img_id}_compare.png")
        cv2.imwrite(save_path, concat_bgr)

    print(f"生成完毕，请到 {cfg.OUTPUT_DIR} 目录下挑选图片。")

if __name__ == "__main__":
    main()
