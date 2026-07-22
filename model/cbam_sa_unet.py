"""
CBAM-SAUNet

参考论文:
1. SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation
   https://arxiv.org/abs/2001.07645
2. CBAM_SAUNet: A novel attention U-Net for effective segmentation of corner cases
   https://doi.org/10.1109/EMBC53108.2024.10782335
3. CBAM: Convolutional Block Attention Module
   https://arxiv.org/abs/1807.06521

实现说明:
- 这版不再是原来那种 “bottleneck 插一个 CBAM” 的组合实现。
- 结构改为贴近论文所述的 Shape Attentive U-Net 路线:
  1. 使用 dual-attention decoder
  2. 保留 gated shape stream 做边界引导
  3. 在 decoder 的 channel attention path 中使用 CBAM 风格通道注意力
- 由于 IEEE 全文未公开到代码层细节，这里按摘要描述将 CBAM 变体落实到
  decoder channel path，上层拓扑对齐到 Shape Attentive U-Net 家族。
"""

import torch

from model.shape_attentive_blocks import CBAMChannelAttention, ShapeAttentiveUNetBase
from model.unet import count_params_flops


class CBAMSAUNet(ShapeAttentiveUNetBase):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_filters=16,
        drop_prob=0.1,
        block_size=7,
        cbam_reduction=16,
        spatial_kernel=7,
    ):
        self.block_size = block_size
        self.spatial_kernel = spatial_kernel
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            drop_prob=drop_prob,
            channel_attention_factory=lambda channels: CBAMChannelAttention(
                channels,
                reduction=cbam_reduction,
            ),
        )


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBAMSAUNet().to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
