"""
CA-SAUNet

参考论文:
1. SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation
   https://arxiv.org/abs/2001.07645
2. CA-SAUNet: Coordinate-Attention Enhanced Shape Attentive UNet for medical
   and agricultural image segmentation
   https://doi.org/10.1007/s10044-025-01590-y
3. Coordinate Attention for Efficient Mobile Network Design
   https://arxiv.org/abs/2103.02907

实现说明:
- 这版结构改为贴近 CA-SAUNet 论文描述，而不是原来在 bottleneck 处
  简单插入 Coordinate Attention。
- 按论文摘要，CA-SAUNet 的核心变化是:
  1. 基于 Shape Attentive U-Net
  2. 保留 gated shape stream
  3. 用 Coordinate Attention 替换 dual-attention decoder 中原有的 SE 模块
"""

import torch

from model.shape_attentive_blocks import CoordinateAttention, ShapeAttentiveUNetBase
from model.unet import count_params_flops


class CASAUNet(ShapeAttentiveUNetBase):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_filters=16,
        drop_prob=0.1,
        block_size=7,
        ca_reduction=32,
    ):
        self.block_size = block_size
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            drop_prob=drop_prob,
            channel_attention_factory=lambda channels: CoordinateAttention(
                channels,
                reduction=ca_reduction,
            ),
        )


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASAUNet().to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
