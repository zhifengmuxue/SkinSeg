"""
CBAM-SAUNet

实现说明:
1. 主干沿用 SA-UNet 的轻量三层 encoder-decoder 与 DropBlock 设计
2. bottleneck 位置不再使用原始 SA-UNet 的 Spatial Attention，
   改为接入 CBAM: Channel Attention -> Spatial Attention
3. 因此这里的 "CBAM-SAUNet" 是一个组合结构实现，
   论文依据分别来自 SA-UNet 与 CBAM 两篇文章

参考论文:
1. SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation
   https://arxiv.org/abs/2004.03696
2. CBAM: Convolutional Block Attention Module
   https://arxiv.org/abs/1807.06521
"""

import torch
import torch.nn as nn

from model.sa_unet import DecoderBlock, EncoderBlock, drop_block_2d
from model.unet import count_params_flops


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力，采用 avg/max pooling + shared MLP。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力，采用通道维 avg/max 聚合后卷积生成空间权重。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_pool, max_pool], dim=1))
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    """按论文顺序依次执行 Channel Attention 和 Spatial Attention。"""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CBAMSAUNet(nn.Module):
    """
    SA-UNet 主干 + CBAM bottleneck.

    说明:
    - encoder / decoder / DropBlock 复用 SA-UNet 论文中的轻量结构思想
    - bottleneck attention 改为 CBAM 论文中的串联通道-空间注意力
    """

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
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.enc1 = EncoderBlock(in_channels, base_filters, drop_prob=drop_prob, block_size=block_size)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, drop_prob=drop_prob, block_size=block_size)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, drop_prob=drop_prob, block_size=block_size)

        self.bridge_conv1 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False)
        self.bridge_bn1 = nn.BatchNorm2d(base_filters * 8)
        self.bridge_attn = CBAM(base_filters * 8, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.bridge_conv2 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1, bias=False)
        self.bridge_bn2 = nn.BatchNorm2d(base_filters * 8)
        self.relu = nn.ReLU(inplace=True)

        self.dec1 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4, drop_prob=drop_prob, block_size=block_size)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2, drop_prob=drop_prob, block_size=block_size)
        self.dec3 = DecoderBlock(base_filters * 2, base_filters, base_filters, drop_prob=drop_prob, block_size=block_size)
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        x = self.bridge_conv1(p3)
        x = drop_block_2d(
            x,
            drop_prob=self.drop_prob,
            block_size=self.block_size,
            training=self.training,
        )
        x = self.relu(self.bridge_bn1(x))
        x = self.bridge_attn(x)
        x = self.bridge_conv2(x)
        x = drop_block_2d(
            x,
            drop_prob=self.drop_prob,
            block_size=self.block_size,
            training=self.training,
        )
        x = self.relu(self.bridge_bn2(x))

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        return self.outc(x)


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBAMSAUNet().to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
