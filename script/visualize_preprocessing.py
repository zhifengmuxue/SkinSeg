"""
论文候选图可视化脚本

功能:
1. 读取 1 个 Ground Truth 样本
2. 加载五个模型的 checkpoint 并执行推理
3. 生成 5 种不同排版的论文候选图，方便挑选放入文章

默认比较的五个模型:
- U-Net
- Attention U-Net
- SA-UNet
- CBAM-SAUNet
- CA-SAUNet
"""

import argparse
import os
import sys
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_val_transforms
from model.factory import build_model

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\code\segment\dataset"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paper_prediction_layouts")
DEFAULT_SAMPLE_IDS = [

    "ISIC_0012647",
]
DEFAULT_SPLIT = "test"
DEFAULT_IMG_SIZE = 256
DEFAULT_DPI = 220

MODEL_SPECS = [
    {
        "name": "U-Net",
        "yaml_path": r"d:\code\segment\config\unet.yml",
        "title_color": "#2E86DE",
    },
    {
        "name": "Attention U-Net",
        "yaml_path": r"d:\code\segment\config\attention_unet.yml",
        "title_color": "#8E44AD",
    },
    {
        "name": "SA-UNet",
        "yaml_path": r"d:\code\segment\config\sa_unet.yml",
        "title_color": "#16A085",
    },
    {
        "name": "CBAM-SAUNet",
        "yaml_path": r"d:\code\segment\config\cbam_sa_unet.yml",
        "title_color": "#D35400",
    },
    {
        "name": "CA-SAUNet",
        "yaml_path": r"d:\code\segment\config\ca_sa_unet.yml",
        "title_color": "#C0392B",
    },
]


def get_split_dirs(split):
    split = split.lower()
    if split == "train":
        img_dir = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Training_Input")
        mask_dir = os.path.join(BASE_DIR, "ISIC2018_Task1_Training_GroundTruth")
    elif split == "val":
        img_dir = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Validation_Input")
        mask_dir = os.path.join(BASE_DIR, "ISIC2018_Task1_Validation_GroundTruth")
    elif split == "test":
        img_dir = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Test_Input")
        mask_dir = os.path.join(BASE_DIR, "ISIC2018_Task1_Test_GroundTruth")
    else:
        raise ValueError(f"不支持的 split: {split}，可选 train / val / test")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"标注目录不存在: {mask_dir}")

    return img_dir, mask_dir


def load_image_and_mask(sample_id, split, img_size):
    img_dir, mask_dir = get_split_dirs(split)
    img_path = os.path.join(img_dir, f"{sample_id}.jpg")
    mask_path = os.path.join(mask_dir, f"{sample_id}_segmentation.png")

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取标注: {mask_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 127).astype(np.uint8) * 255
    return rgb, rgb_resized, mask_bin


def make_overlay(rgb_img, mask_bin, fill_color, contour_color, alpha=0.45):
    overlay = rgb_img.copy()
    region = mask_bin > 0
    overlay[region] = (
        overlay[region] * (1.0 - alpha) + np.array(fill_color) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, contour_color, 2)
    return overlay


def build_runtime_cfg(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    predict_cfg = raw_cfg.get("predict", {})
    device_str = raw_cfg.get("device", "cuda")
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and device_str == "cuda"
        else torch.device("cpu")
    )

    return SimpleNamespace(
        DEVICE=device,
        DEVICE_STR=device_str,
        MODEL_TYPE=raw_cfg.get("model_type", "unet"),
        IN_CHANNELS=raw_cfg.get("in_channels", 3),
        OUT_CHANNELS=raw_cfg.get("out_channels", 1),
        BASE_FILTERS=raw_cfg.get("base_filters", 64),
        BILINEAR=raw_cfg.get("bilinear", True),
        THRESHOLD=raw_cfg.get("threshold", 0.5),
        MODEL_KWARGS=raw_cfg.get("model_kwargs", {}) or {},
        RUN_DIR=predict_cfg.get("run_dir", ""),
        MODEL_PATH=predict_cfg.get("model_path", ""),
    )


def resolve_model_path(cfg):
    if cfg.MODEL_PATH:
        return cfg.MODEL_PATH

    if not cfg.RUN_DIR:
        raise FileNotFoundError("配置中未提供 MODEL_PATH，也未提供 RUN_DIR。")

    best_path = os.path.join(cfg.RUN_DIR, "best.pth")
    last_path = os.path.join(cfg.RUN_DIR, "last.pth")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path

    raise FileNotFoundError(
        "未找到模型权重，请确认以下路径至少存在一个:\n"
        f"  {best_path}\n"
        f"  {last_path}"
    )


def load_model_entry(spec):
    cfg = build_runtime_cfg(spec["yaml_path"])
    model_path = resolve_model_path(cfg)
    model = build_model(cfg)

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "name": spec["name"],
        "title_color": spec["title_color"],
        "cfg": cfg,
        "model": model,
        "model_path": model_path,
    }


@torch.no_grad()
def predict_mask(model_entry, raw_rgb, img_size):
    transform = get_val_transforms(img_size)
    tensor = transform(image=raw_rgb)["image"].unsqueeze(0).to(model_entry["cfg"].DEVICE)
    logits = model_entry["model"](tensor)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_mask = (probs > model_entry["cfg"].THRESHOLD).astype(np.uint8) * 255
    return pred_mask


def add_panel(
    ax,
    image,
    title=None,
    title_color="#2C3E50",
    border_color="#D5DBDB",
    show_title=True,
    show_border=True,
):
    ax.imshow(image)
    if show_title and title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(show_border)
        if show_border:
            spine.set_linewidth(2)
            spine.set_edgecolor(border_color)


def build_panels(sample_id, split, img_size, model_entries):
    raw_rgb, resized_rgb, gt_mask = load_image_and_mask(sample_id, split, img_size)
    gt_overlay = make_overlay(
        resized_rgb,
        gt_mask,
        fill_color=(80, 220, 120),
        contour_color=(0, 255, 0),
    )

    panels = [
        {
            "title": "Ground Truth",
            "image": gt_overlay,
            "title_color": "#27AE60",
            "border_color": "#7DCEA0",
        }
    ]

    for entry in model_entries:
        pred_mask = predict_mask(entry, raw_rgb, img_size)
        pred_overlay = make_overlay(
            resized_rgb,
            pred_mask,
            fill_color=(255, 80, 80),
            contour_color=(255, 255, 0),
        )
        panels.append(
            {
                "title": entry["name"],
                "image": pred_overlay,
                "title_color": entry["title_color"],
                "border_color": "#D6DBDF",
            }
        )

    return panels


def finalize_figure(fig, save_path, dpi):
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def render_layout_1_strip(panels, sample_id, split, output_dir, dpi):
    fig = plt.figure(figsize=(18, 3.1), facecolor="white")
    gs = fig.add_gridspec(1, 6, left=0.001, right=0.999, top=0.999, bottom=0.001, wspace=0.01)
    for idx, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        add_panel(
            ax,
            panel["image"],
            title=None,
            title_color=panel["title_color"],
            border_color=panel["border_color"],
            show_title=False,
            show_border=False,
        )

    save_path = os.path.join(output_dir, f"{sample_id}_layout_01_clean.png")
    finalize_figure(fig, save_path, dpi)


def render_layout_2_grid(panels, sample_id, split, output_dir, dpi):
    fig = plt.figure(figsize=(12.5, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.96, top=0.90, bottom=0.08, hspace=0.18, wspace=0.08)
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        add_panel(ax, panel["image"], panel["title"], panel["title_color"], panel["border_color"])

    save_path = os.path.join(output_dir, f"{sample_id}_layout_02_grid.png")
    finalize_figure(fig, save_path, dpi)


def render_layout_3_compact(panels, sample_id, split, output_dir, dpi):
    fig = plt.figure(figsize=(9.5, 13.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, left=0.06, right=0.94, top=0.92, bottom=0.05, hspace=0.16, wspace=0.08)
    order = [0, 1, 2, 3, 4, 5]
    for slot, panel_idx in enumerate(order):
        row, col = divmod(slot, 2)
        panel = panels[panel_idx]
        ax = fig.add_subplot(gs[row, col])
        add_panel(ax, panel["image"], panel["title"], panel["title_color"], panel["border_color"])

    save_path = os.path.join(output_dir, f"{sample_id}_layout_03_compact.png")
    finalize_figure(fig, save_path, dpi)


def render_layout_4_hero_gt(panels, sample_id, split, output_dir, dpi):
    fig = plt.figure(figsize=(14.5, 14.5), facecolor="white")
    gs = fig.add_gridspec(
        5,
        2,
        left=0.04,
        right=0.96,
        top=0.93,
        bottom=0.04,
        hspace=0.10,
        wspace=0.08,
        width_ratios=[1.25, 1.0],
    )

    gt_panel = panels[0]
    ax_gt = fig.add_subplot(gs[:, 0])
    add_panel(ax_gt, gt_panel["image"], gt_panel["title"], gt_panel["title_color"], gt_panel["border_color"])

    for idx in range(1, len(panels)):
        panel = panels[idx]
        ax = fig.add_subplot(gs[idx - 1, 1])
        add_panel(ax, panel["image"], panel["title"], panel["title_color"], panel["border_color"])

    save_path = os.path.join(output_dir, f"{sample_id}_layout_04_hero_gt.png")
    finalize_figure(fig, save_path, dpi)


def render_layout_5_gt_top(panels, sample_id, split, output_dir, dpi):
    fig = plt.figure(figsize=(18, 9.8), facecolor="white")
    gs = fig.add_gridspec(
        2,
        5,
        left=0.03,
        right=0.97,
        top=0.91,
        bottom=0.07,
        hspace=0.16,
        wspace=0.06,
        height_ratios=[1.15, 1.0],
    )

    gt_panel = panels[0]
    ax_gt = fig.add_subplot(gs[0, :])
    add_panel(ax_gt, gt_panel["image"], gt_panel["title"], gt_panel["title_color"], gt_panel["border_color"])

    for idx in range(1, len(panels)):
        panel = panels[idx]
        ax = fig.add_subplot(gs[1, idx - 1])
        add_panel(ax, panel["image"], panel["title"], panel["title_color"], panel["border_color"])

    save_path = os.path.join(output_dir, f"{sample_id}_layout_05_gt_top.png")
    finalize_figure(fig, save_path, dpi)


def render_all_layouts(panels, sample_id, split, output_dir, dpi):
    render_layout_1_strip(panels, sample_id, split, output_dir, dpi)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成包含 Ground Truth 与五个模型预测结果的五种论文候选图。"
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=DEFAULT_SAMPLE_IDS,
        help="要可视化的样本 ID，可同时传多个；默认会批量生成多张 layout1 纯图。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test"],
        help="数据划分，默认 val。",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="推理与绘图使用的图像尺寸，默认 256。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认保存到 script/paper_prediction_layouts。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="输出分辨率 DPI，默认 220。",
    )
    return parser.parse_args()


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("开始加载五个模型...")
    model_entries = [load_model_entry(spec) for spec in MODEL_SPECS]
    for entry in model_entries:
        print(f"  - {entry['name']}: {entry['model_path']}")

    for sample_id in args.sample_ids:
        print(f"\n开始生成样本 {sample_id} 的候选图...")
        panels = build_panels(sample_id, args.split, args.img_size, model_entries)
        render_all_layouts(panels, sample_id, args.split, args.output_dir, args.dpi)
        print(f"已完成: {args.output_dir}")

    print("\n全部候选图生成完毕。")


if __name__ == "__main__":
    main()
