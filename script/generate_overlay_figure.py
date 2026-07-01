"""
生成 ISIC 2018 皮肤病变分割 — 3×2 叠加对比图：
左列：原始皮肤镜图像   |   右列：mask 叠加在原图上
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\code\segment\dataset"
IMG_DIR   = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Validation_Input")
MASK_DIR  = os.path.join(BASE_DIR, "ISIC2018_Task1_Validation_GroundTruth")
OUTPUT_PATH = os.path.join(r"d:\code\segment", "overlay_figure.png")

# 选 3 个验证集样本
SAMPLE_IDS = ["ISIC_0012633", "ISIC_0012255", "ISIC_0020233"]

# ============================================================
# 图像处理
# ============================================================
def load_pair(sample_id):
    img_path = os.path.join(IMG_DIR, f"{sample_id}.jpg")
    mask_path = os.path.join(MASK_DIR, f"{sample_id}_segmentation.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype(np.uint8) * 255
    return img, mask_bin


def create_overlay(img, mask_bin, alpha=0.45, color=(255, 80, 80)):
    overlay = img.copy()
    colored_mask = np.zeros_like(img)
    colored_mask[:, :, 0] = mask_bin // 255 * color[0]
    colored_mask[:, :, 1] = mask_bin // 255 * color[1]
    colored_mask[:, :, 2] = mask_bin // 255 * color[2]
    blended = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)
    return blended


# ============================================================
# 主绘图
# ============================================================
def main():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(14, 18), facecolor='white')
    fig.suptitle("皮肤镜图像 — 病灶分割 Mask 叠加对比",
                 fontsize=22, fontweight='bold', y=0.98, color='#2C3E50')

    for i, sid in enumerate(SAMPLE_IDS):
        img, mask_bin = load_pair(sid)
        overlay = create_overlay(img, mask_bin)

        # 左列：原始图像
        ax_left = fig.add_subplot(3, 2, i * 2 + 1)
        ax_left.imshow(img)
        ax_left.set_title(f"原始图像 ({sid})", fontsize=13,
                          fontweight='bold', color='#2C3E50', pad=8)
        ax_left.axis('off')

        # 右列：mask 叠加
        ax_right = fig.add_subplot(3, 2, i * 2 + 2)
        ax_right.imshow(overlay)
        ax_right.set_title(f"分割 Mask 叠加 ({sid})", fontsize=13,
                           fontweight='bold', color='#2C3E50', pad=8)
        ax_right.axis('off')

    # 图例
    legend_patches = [
        mpatches.Patch(color='#FF5050', alpha=0.45, label='病灶区域'),
        mpatches.Patch(facecolor='none', edgecolor='yellow', linewidth=2, label='病灶轮廓'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               framealpha=0.9, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    # 底部说明
    fig.text(0.5, 0.018, "数据来源: ISIC 2018 Challenge — Task 1 Lesion Segmentation (Validation Set)",
             ha='center', fontsize=10, color='#95A5A6', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[叠加对比图] 已保存至: {OUTPUT_PATH}")
    print(f"样本 ID: {SAMPLE_IDS}")


if __name__ == "__main__":
    main()
