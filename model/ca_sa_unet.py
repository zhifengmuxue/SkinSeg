"""
CA-SAUNet

实现说明:
1. 主干沿用 SA-UNet 的轻量三层 encoder-decoder 与 DropBlock 设计
2. bottleneck 位置不再使用原始 SA-UNet 的 Spatial Attention，
   改为接入 Coordinate Attention
3. 因此这里的 "CA-SAUNet" 是一个组合结构实现，
   论文依据分别来自 SA-UNet 与 Coordinate Attention 两篇文章

参考论文:
1. SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation
   https://arxiv.org/abs/2004.03696
2. Coordinate Attention for Efficient Mobile Network Design
   https://arxiv.org/abs/2103.02907
"""

import torch
import torch.nn as nn

from model.sa_unet import DecoderBlock, EncoderBlock, drop_block_2d
from model.unet import count_params_flops


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention.

    论文核心思路:
    - 分别沿 H/W 方向做一维全局平均池化，保留位置信息
    - 共享 1x1 变换后拆分成两个分支，分别生成高度与宽度注意力
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class CASAUNet(nn.Module):
    """
    SA-UNet 主干 + Coordinate Attention bottleneck.

    说明:
    - encoder / decoder / DropBlock 复用 SA-UNet 论文中的轻量结构思想
    - bottleneck attention 改为 Coordinate Attention 论文中的坐标注意力
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_filters=16,
        drop_prob=0.1,
        block_size=7,
        ca_reduction=32,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.enc1 = EncoderBlock(in_channels, base_filters, drop_prob=drop_prob, block_size=block_size)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, drop_prob=drop_prob, block_size=block_size)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, drop_prob=drop_prob, block_size=block_size)

        self.bridge_conv1 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False)
        self.bridge_bn1 = nn.BatchNorm2d(base_filters * 8)
        self.bridge_attn = CoordinateAttention(base_filters * 8, reduction=ca_reduction)
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
    model = CASAUNet().to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
