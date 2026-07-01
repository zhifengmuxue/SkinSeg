"""
数据预处理流程可视化

生成 ISIC 2018 皮肤病变分割项目的预处理流水线示意图。
使用与 data/dataset.py 完全一致的 albumentations 增强方法，
展示每种变换的图像效果及其对应的 mask (叠加到图像上)。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

import albumentations as A

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\code\segment\dataset"
SAMPLE_ID = "ISIC_0012633"
IMG_PATH = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Validation_Input", f"{SAMPLE_ID}.jpg")
MASK_PATH = os.path.join(BASE_DIR, "ISIC2018_Task1_Validation_GroundTruth",
                         f"{SAMPLE_ID}_segmentation.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "preprocessing_pipeline.png")

IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# ============================================================
# 增强定义 — 与 data/dataset.py get_train_transforms 完全一致
# ============================================================
AUG_CATEGORIES = [
    ("随机裁剪缩放",
     A.Compose([A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0),
                                     ratio=(0.9, 1.1), p=1.0)])),
    ("水平/垂直翻转",
     A.Compose([
         A.HorizontalFlip(p=1.0),
         A.VerticalFlip(p=1.0),
     ])),
    ("旋转",
     A.Compose([A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0)])),
    ("亮度/对比度",
     A.Compose([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)])),
    ("高斯噪声",
     A.Compose([A.GaussNoise(std_range=(0.02, 0.08), p=1.0)])),
    ("高斯模糊",
     A.Compose([A.GaussianBlur(blur_limit=(3, 5), p=1.0)])),
    ("弹性变形",
     A.Compose([A.ElasticTransform(alpha=1, sigma=50, p=1.0)])),
]


def make_overlay(img, mask):
    """mask 叠加到原图：红色半透明 + 黄色轮廓 (与 GT mask 样式一致)"""
    mask_bin = (mask > 127).astype(np.uint8)
    overlay = img.copy()
    overlay[mask_bin > 0] = (
        overlay[mask_bin > 0] * 0.5 +
        np.array([255, 80, 80]) * 0.5
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay


# ============================================================
# 绘制流程图组件
# ============================================================
def draw_flow_box(ax, x, y, w, h, text, subtitle=None, color='#3498DB'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + 2, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 12, subtitle, ha='center', va='center',
                fontsize=7, color='#E0E0E0')


def draw_arrow(ax, x1, y1, x2, y2, color='#7F8C8D'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                connectionstyle='arc3,rad=0'))


# ============================================================
# 主绘图
# ============================================================
def main():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # ---- 加载原始数据 ----
    raw = cv2.imread(IMG_PATH)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    mask_raw = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # ---- 归一化 (可视化用逆归一化) ----
    img_f = resized.astype(np.float32) / 255.0
    normalized = (img_f - MEAN) / STD
    normalized_vis = np.clip(normalized * STD + MEAN, 0, 1)

    # ---- GT Mask 叠加 ----
    gt_overlay = make_overlay(resized, mask_resized)

    # ---- 用 albumentations 生成每种增强的结果 (image + mask 同步) ----
    aug_pairs = []  # [(label, aug_img_uint8, aug_mask_overlay_uint8)]
    for label, transform in AUG_CATEGORIES:
        data = transform(image=resized, mask=mask_resized)
        aug_img = data['image'].astype(np.uint8)
        aug_mask = data['mask']
        aug_overlay = make_overlay(aug_img, aug_mask)
        aug_pairs.append((label, aug_img, aug_overlay))

    # ============================================================
    # 创建图表
    # ============================================================
    fig = plt.figure(figsize=(28, 18), facecolor='white')

    # ---- 顶部流程图 ----
    ax_flow = fig.add_axes([0.02, 0.80, 0.96, 0.15])
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 30)
    ax_flow.axis('off')
    ax_flow.text(50, 28, "ISIC 2018 皮肤病变分割 — 数据预处理流水线",
                 ha='center', fontsize=16, fontweight='bold', color='#2C3E50')

    box_w, box_h = 14, 8
    y_box = 12
    gap = 2
    colors = ['#3498DB', '#2ECC71', '#E67E22', '#9B59B6', '#E74C3C']
    steps = [
        ("原始图像", "600x450 ~ 4288x2848"),
        ("Resize", "256 x 256"),
        ("归一化", "ImageNet mean/std"),
        ("数据增强", "Flip / Color / Noise / Elastic"),
        ("模型输入", "Tensor (3,256,256)"),
    ]
    for i, (title, sub) in enumerate(steps):
        x = 3 + i * (box_w + gap)
        draw_flow_box(ax_flow, x, y_box, box_w, box_h, title, sub, colors[i])
        if i < len(steps) - 1:
            nx = 3 + (i + 1) * (box_w + gap)
            draw_arrow(ax_flow, x + box_w + 0.5, y_box + box_h / 2,
                       nx - 0.5, y_box + box_h / 2)

    ax_flow.text(50, 3,
                 "增强策略: RandomResizedCrop → Flip → Rotate → ColorJitter → "
                 "GaussNoise/GaussianBlur → ElasticTransform → Normalize",
                 ha='center', fontsize=9, color='#888')

    # ---- 下半部分 ----
    # 3 行: Row0(原始/Resize/归一化/GT), Row1(增强-图像), Row2(增强-Mask叠加)
    n_aug = len(AUG_CATEGORIES)  # 7
    gs = fig.add_gridspec(3, 8, left=0.02, right=0.98, top=0.78, bottom=0.04,
                          hspace=0.32, wspace=0.06)

    # Row 0: 步骤 ①②③ + GT Mask
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(raw)
    ax.set_title("① 原始皮肤镜图像", fontsize=11, fontweight='bold', color='#3498DB', pad=6)
    ax.text(raw.shape[1]/2, raw.shape[0]+20, f"{raw.shape[1]} x {raw.shape[0]}",
            ha='center', fontsize=8, color='#888')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2:4])
    ax.imshow(resized)
    ax.set_title("② Resize -> 256x256", fontsize=11, fontweight='bold', color='#2ECC71', pad=6)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 4:6])
    ax.imshow(normalized_vis)
    ax.set_title("③ 归一化 (ImageNet)", fontsize=11, fontweight='bold', color='#E67E22', pad=6)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 6:8])
    ax.imshow(gt_overlay)
    ax.set_title("标注 Mask (Ground Truth)", fontsize=11, fontweight='bold', color='#E74C3C', pad=6)
    ax.axis('off')

    # Row 1: ④ 数据增强 — 图像
    ax_img = fig.add_subplot(gs[1, :])
    ax_img.set_title("④ 数据增强 (Image)",
                     fontsize=11, fontweight='bold', color='#9B59B6', pad=6)
    ax_img.axis('off')
    for i, (label, aug_img, _) in enumerate(aug_pairs):
        left = i / n_aug
        w = 1.0 / n_aug
        inset = ax_img.inset_axes([left + 0.003, 0.05, w - 0.006, 0.88])
        inset.imshow(aug_img)
        inset.set_title(label, fontsize=8, color='#555', pad=3)
        inset.axis('off')

    # Row 2: ⑤ Mask 叠加到增强图像上 (与 GT mask 风格统一)
    ax_mask = fig.add_subplot(gs[2, :])
    # ax_mask.set_title("⑤ 数据增强 — Mask 叠加 (Mask overlaid on Image)",
    #                   fontsize=11, fontweight='bold', color='#C0392B', pad=6)
    ax_mask.axis('off')
    for i, (label, _, aug_overlay) in enumerate(aug_pairs):
        left = i / n_aug
        w = 1.0 / n_aug
        inset = ax_mask.inset_axes([left + 0.003, 0.05, w - 0.006, 0.88])
        inset.imshow(aug_overlay)
        inset.set_title(label, fontsize=8, color='#555', pad=3)
        inset.axis('off')

    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"预处理流程可视化图已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
