"""
生成 ISIC 2018 皮肤病变分割分析流水线组合图：
原始皮肤镜图像 → 病灶分割 mask → 后续分类/诊断辅助
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\code\segment\dataset"
SAMPLE_ID = "ISIC_0012633"  # 验证集中有原图和 mask 的病例

IMG_PATH = os.path.join(BASE_DIR, "ISIC2018_Task1-2_Validation_Input", f"{SAMPLE_ID}.jpg")
MASK_PATH = os.path.join(BASE_DIR, "ISIC2018_Task1_Validation_GroundTruth", f"{SAMPLE_ID}_segmentation.png")
OUTPUT_PATH = os.path.join(r"d:\code\segment", "pipeline_figure.png")

# ISIC 2018 七分类标签
DISEASE_CLASSES = [
    ("MEL",   "黑色素瘤 (Melanoma)",           "#E74C3C"),
    ("NV",    "色素痣 (Melanocytic Nevus)",     "#3498DB"),
    ("BCC",   "基底细胞癌 (BCC)",              "#E67E22"),
    ("AKIEC", "光化性角化病/Bowen (AKIEC)",    "#9B59B6"),
    ("BKL",   "良性角化病 (BKL)",              "#2ECC71"),
    ("DF",    "皮肤纤维瘤 (DF)",                "#1ABC9C"),
    ("VASC",  "血管病变 (VASC)",                "#F39C12"),
]

# ============================================================
# 图像处理
# ============================================================
def load_and_process():
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
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
# Panel 3: 分类/诊断辅助 (matplotlib 原生绘制)
# ============================================================
def draw_classification_panel(ax, img):
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.invert_yaxis()
    ax.set_facecolor('#F5F5FA')
    ax.axis('off')

    # 顶部标题栏
    ax.add_patch(plt.Rectangle((0, 0), 500, 52, facecolor='#2C3E50', edgecolor='none'))
    ax.text(18, 12, "后续分类 / 诊断辅助", fontsize=12, color='white',
            fontweight='bold', va='top')
    ax.text(18, 32, "Lesion Classification & Diagnosis", fontsize=8,
            color='#B4C8DC', va='top')

    # 左侧病灶缩略图
    thumb = Image.fromarray(img).resize((90, 90), Image.LANCZOS)
    thumb_arr = np.array(thumb)
    ax.add_patch(plt.Rectangle((16, 66), 94, 94, facecolor='#C8C8C8', edgecolor='none'))
    ax.imshow(thumb_arr, extent=[18, 108, 162, 72], aspect='auto')

    # 右侧特征提取
    ax.text(130, 76, "病灶特征提取:", fontsize=10, color='#2C3E50', fontweight='bold')
    features = [
        "－ 不对称性 (Asymmetry)",
        "－ 边界不规则 (Border)",
        "－ 颜色多样性 (Color)",
        "－ 直径 (Diameter)",
        "－ 纹理特征 (Texture)",
        "－ 深层特征 (Deep Features)",
    ]
    for i, feat in enumerate(features):
        ax.text(135, 94 + i * 16, feat, fontsize=9, color='#555')

    # 分隔线
    ax.axhline(y=185, xmin=0.03, xmax=0.97, color='#C8C8D2', linewidth=1.5)

    # 分类器区域
    rect_y = 195
    ax.add_patch(plt.Rectangle((15, rect_y), 470, 135, fill=False,
                                edgecolor='#B4B4BE', linewidth=1))
    ax.plot(28, rect_y + 12, 'o', color='#2E8B57', markersize=8)
    ax.text(46, rect_y + 12, "CNN 分类器 (ResNet / EfficientNet)", fontsize=9,
            color='#2C3E50', fontweight='bold', va='center')

    # 7 类柱状图
    probs = [0.72, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
    bar_start_x = 22
    bar_w = 56
    bar_gap = 6
    bar_base_y = rect_y + 130
    bar_max = 100
    for i, (abbr, _, color) in enumerate(DISEASE_CLASSES):
        bx = bar_start_x + i * (bar_w + bar_gap)
        bh = probs[i] * bar_max
        ax.add_patch(plt.Rectangle((bx, bar_base_y - bh), bar_w, bh,
                                    facecolor=color, edgecolor='white', linewidth=1))
        ax.text(bx + bar_w / 2, bar_base_y + 5, abbr, fontsize=7,
                ha='center', color='#666')

    # 诊断结论
    diag_y = 350
    ax.add_patch(plt.Rectangle((15, diag_y), 470, 42, fill=False,
                                edgecolor='#E74C3C', linewidth=2))
    ax.text(25, diag_y + 4, "诊断结论:", fontsize=10, color='#E74C3C',
            fontweight='bold', va='top')
    ax.text(25, diag_y + 22, "黑色素瘤 (Melanoma)  —  高置信度 (72%)",
            fontsize=10, color='#2C3E50', va='top')

    # 底部提示
    ax.text(15, 405, "建议: 结合皮肤镜检查与病理活检综合判断，AI 辅助仅供参考",
            fontsize=8, color='#A0A0A0')


# ============================================================
# 主绘图
# ============================================================
def main():
    img, mask_bin = load_and_process()
    overlay = create_overlay(img, mask_bin)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 7.5), facecolor='white')
    fig.suptitle("皮肤镜图像病灶分割与辅助诊断流水线",
                 fontsize=20, fontweight='bold', y=0.98, color='#2C3E50')

    # ---- Panel 1: 原始图像 ----
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title("原始皮肤镜图像\n(Dermoscopic Image)", fontsize=14,
                  fontweight='bold', color='#2C3E50', pad=12)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((-8, -8), img.shape[1] + 16, img.shape[0] + 16,
                                fill=False, edgecolor='#3498DB', linewidth=3, linestyle='--'))

    # ---- Panel 2: 分割 Mask ----
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(overlay)
    ax2.set_title("病灶分割 Mask\n(Segmentation Mask)", fontsize=14,
                  fontweight='bold', color='#2C3E50', pad=12)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((-8, -8), img.shape[1] + 16, img.shape[0] + 16,
                                fill=False, edgecolor='#E74C3C', linewidth=3, linestyle='--'))

    legend_patches = [
        mpatches.Patch(color='#FF5050', alpha=0.45, label='病灶区域'),
        mpatches.Patch(facecolor='none', edgecolor='yellow', linewidth=2, label='病灶轮廓'),
    ]
    ax2.legend(handles=legend_patches, loc='lower center', ncol=2,
               framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    # ---- Panel 3: 分类/诊断 ----
    ax3 = fig.add_subplot(1, 3, 3)
    draw_classification_panel(ax3, img)

    # ---- 箭头 ----
    kw_arrow = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=6",
                    color='#7F8C8D', lw=2.0)
    fig.add_artist(FancyArrowPatch((0.335, 0.52), (0.345, 0.52),
                                    transform=fig.transFigure, **kw_arrow))
    fig.add_artist(FancyArrowPatch((0.668, 0.52), (0.678, 0.52),
                                    transform=fig.transFigure, **kw_arrow))

    # 底部说明
    fig.text(0.5, 0.01,
             "数据来源: ISIC 2018 Challenge — Task 1 (Lesion Segmentation) & Task 3 (Disease Classification)",
             ha='center', fontsize=10, color='#95A5A6', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[流水线图] 已保存至: {OUTPUT_PATH}")
    print(f"样本 ID: {SAMPLE_ID}")


if __name__ == "__main__":
    main()
