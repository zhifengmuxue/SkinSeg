# SkinSeg

基于 PyTorch 的皮肤镜图像病灶分割框架，支持 6 种 U-Net 变体，覆盖 ISIC 2018 Challenge Task 1 全流程：训练、评估、推理与可视化。

## 支持的模型

| 模型 | 论文 | 核心改进 |
|------|------|----------|
| **U-Net** | Ronneberger et al. 2015 | 经典编码器-解码器 + Skip Connection |
| **Attention U-Net** | Oktay et al. 2018 | Additive Attention Gate |
| **SA-UNet** | 2020 | Dual-Attention Decoder + Gated Shape Stream |
| **CBAM-SAUNet** | EMBC 2024 | CBAM 通道注意力替换 SE |
| **CA-SAUNet** | 2025 | Coordinate Attention 替换 SE |
| **WGS-SAUNet** | 本工作 | Haar DWT 小波先验注入 Shape Stream |

## 目录结构

```
segment/
├── config/                        # 模型配置
│   ├── __init__.py                # Config 统一加载器
│   ├── config.yml                 # 默认配置 (U-Net)
│   ├── unet.yml
│   ├── attention_unet.yml
│   ├── sa_unet.yml
│   ├── cbam_sa_unet.yml
│   ├── ca_sa_unet.yml
│   └── wgs_sa_unet.yml
├── data/
│   └── dataset.py                 # ISIC 数据集 + albumentations 增强
├── model/
│   ├── factory.py                 # 模型工厂，根据 model_type 自动构建
│   ├── unet.py                    # U-Net
│   ├── attention_unet.py          # Attention U-Net
│   ├── shape_attentive_blocks.py  # SAUNet 系列共用模块
│   ├── sa_unet.py                 # Shape Attentive U-Net
│   ├── cbam_sa_unet.py            # CBAM-SAUNet
│   ├── ca_sa_unet.py              # CA-SAUNet
│   └── wgs_sa_unet.py             # WGS-SAUNet
├── script/
│   ├── visualize_preprocessing.py # 论文多模型对比候选图
│   ├── generate_overlay_figure.py # 叠加对比图
│   └── generate_pipeline_figure.py# 流水线组合图
├── train.py                       # 训练
├── eval.py                        # 评估 (Acc / Dice / mIoU)
├── predict.py                     # 推理 (批量/单张)
├── export_visuals.py              # 导出测试集可视化对比
├── losses.py                      # BCEDiceLoss
├── dataset/                       # ISIC 2018 数据集 (gitignored)
└── runs/                          # 训练产物 (gitignored)
```

## 环境安装

```bash
pip install torch torchvision albumentations opencv-python tensorboard tqdm torchinfo matplotlib
```

## 数据准备

下载 [ISIC 2018 Challenge Task 1](https://challenge.isic-archive.com/task/1/) 数据集，按以下结构放置：

```
dataset/
├── ISIC2018_Task1-2_Training_Input/          # 训练原图 (.jpg)
├── ISIC2018_Task1_Training_GroundTruth/      # 训练标注 (_segmentation.png)
├── ISIC2018_Task1-2_Validation_Input/        # 验证原图
├── ISIC2018_Task1_Validation_GroundTruth/    # 验证标注
├── ISIC2018_Task1-2_Test_Input/              # 测试原图
└── ISIC2018_Task1_Test_GroundTruth/          # 测试标注 (如无则跳过)
```

在 `config/config.yml` 中修改 `data_dir` 指向数据集根目录。

## 使用方法

所有命令在项目根目录下执行，通过设置环境变量切换模型：

```bash
# 切换到 U-Net
export SEGMENT_CONFIG=config.yml

# 切换到 CA-SAUNet
export SEGMENT_CONFIG=ca_sa_unet.yml

# 切换到 WGS-SAUNet
export SEGMENT_CONFIG=wgs_sa_unet.yml
```

或直接使用各模型对应的 yml 文件（无需 export）。

### 训练

```bash
python train.py
```

- 断点续训：在 `config.yml` 中设置 `resume` 路径，或自动检测 `runs/<run_name>/last.pth`
- 检查点保存 `best.pth`（最佳验证 Dice）和 `last.pth`（最新 epoch）
- TensorBoard 日志：`runs/<run_name>/tensorboard/`

### 评估

```bash
python eval.py
```

输出指标：**Acc / Dice / mIoU** 以及 **Params / FLOPs**。

### 推理

```bash
python predict.py
```

- 默认对测试集批量推理，保存二值 mask + 概率图 + 叠加图
- 修改 `predict.input_mode` 为 `"file"` 可对单张图像推理

### 导出可视化

```bash
python export_visuals.py
```

随机抽取 N 个测试样本，生成 [原图 | GT 叠加 | 预测叠加] 三栏对比图。

### 论文候选图生成

```bash
# 多模型对比 (5 种模型同一样本对比)
python script/visualize_preprocessing.py --sample-ids ISIC_0012647

# 叠加对比图
python script/generate_overlay_figure.py

# 流水线图
python script/generate_pipeline_figure.py
```

## 配置说明

每个模型都有独立的 yml 配置文件，通过修改 `model_type` 切换架构：

```yaml
model_type: "unet"    # unet / attention_unet / sa_unet / cbam_sa_unet / ca_sa_unet / wgs_sa_unet
base_filters: 64      # 基础通道数，SAUNet 系列建议 16
img_size: 256         # 输入分辨率
```

训练超参数：

```yaml
train:
  batch_size: 8
  num_epochs: 80
  learning_rate: 0.0001
  lr_scheduler: "cosine"   # cosine / step / plateau
  use_amp: true            # 混合精度
  bce_weight: 0.5
  dice_weight: 0.5
```

## 技术要点

- **损失函数**：BCEWithLogitsLoss + Dice Loss 加权组合，兼容 AMP
- **数据增强**：RandomResizedCrop、翻转、旋转、颜色抖动、噪声、弹性变形（albumentations 同步变换 image 和 mask）
- **评估指标**：IoU / Dice / Accuracy
- **Gated Shape Stream**：SAUNet 系列独有的边界引导通路，使用可学习空间门控
- **WGS-SAUNet**：基于固定 Haar DWT 提取高频子带，编码为边界先验注入 Shape Stream

## 引用

```bibtex
@article{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={arXiv preprint arXiv:1505.04597},
  year={2015}
}
```

## License

MIT
