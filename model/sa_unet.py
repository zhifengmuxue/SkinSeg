"""
SAUNet

参考论文:
1. SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation
   https://arxiv.org/abs/2001.07645

实现说明:
- 这里的 `SAUNet` 统一到 Shape Attentive U-Net 路线，作为
  `CBAM-SAUNet / CA-SAUNet` 的共同基线。
- 核心结构包括:
  1. dual-attention decoder
  2. gated shape stream
  3. decoder 中原始的 SE 通道注意力
- 保留你现有项目的接口参数，便于通过 `config/factory` 无缝切换。
"""

import torch

from model.shape_attentive_blocks import SEAttention, ShapeAttentiveUNetBase
from model.unet import count_params_flops


class SAUNet(ShapeAttentiveUNetBase):
    """
    Shape Attentive U-Net 基线版本。

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        base_filters: 基础通道
        drop_prob: 轻量正则化丢弃概率
        block_size: 为兼容旧配置保留，当前实现未显式使用
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=16, drop_prob=0.1, block_size=7):
        self.block_size = block_size
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            drop_prob=drop_prob,
            channel_attention_factory=lambda channels: SEAttention(channels, reduction=16),
        )


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAUNet(in_channels=3, out_channels=1, base_filters=16).to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))

    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
