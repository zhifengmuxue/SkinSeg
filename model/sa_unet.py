"""
SA-UNet

基于论文:
"SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation"
https://arxiv.org/abs/2004.03696

实现要点:
1. 采用更轻量的三层 encoder-decoder
2. 在卷积块中加入 structured dropout (DropBlock)
3. 在 bottleneck 位置加入 spatial attention module
4. 输出保持为 logits，便于与现有训练流程兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet import count_params_flops


def drop_block_2d(x, drop_prob=0.1, block_size=7, training=False):
    """简化版 DropBlock2D，训练时按空间块丢弃局部特征"""
    if (not training) or drop_prob <= 0.0:
        return x

    n, c, h, w = x.shape
    block_size = min(block_size, h, w)
    if block_size < 1:
        return x

    gamma = drop_prob * h * w / (block_size ** 2 * max((h - block_size + 1) * (w - block_size + 1), 1))

    noise = torch.empty((n, c, h - block_size + 1, w - block_size + 1), dtype=x.dtype, device=x.device)
    noise.bernoulli_(gamma)
    noise = F.pad(noise, [block_size // 2] * 4, value=0)
    noise = F.max_pool2d(noise, kernel_size=block_size, stride=1, padding=block_size // 2)
    keep_mask = 1.0 - noise

    normalize_scale = keep_mask.numel() / keep_mask.sum().clamp_min(1.0)
    return x * keep_mask * normalize_scale


class SpatialAttention(nn.Module):
    """沿空间维度生成 attention map"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))
        return x * attn


class SAConvBlock(nn.Module):
    """Conv-BN-ReLU x2，并在每个卷积后应用 DropBlock"""

    def __init__(self, in_channels, out_channels, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = drop_block_2d(x, drop_prob=self.drop_prob, block_size=self.block_size, training=self.training)
        x = self.relu(self.bn1(x))

        x = self.conv2(x)
        x = drop_block_2d(x, drop_prob=self.drop_prob, block_size=self.block_size, training=self.training)
        x = self.relu(self.bn2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, block_size=7):
        super().__init__()
        self.conv = SAConvBlock(in_channels, out_channels, drop_prob=drop_prob, block_size=block_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, drop_prob=0.1, block_size=7):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SAConvBlock(out_channels + skip_channels, out_channels, drop_prob=drop_prob, block_size=block_size)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SAUNet(nn.Module):
    """
    贴近论文的轻量级 SA-UNet

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        base_filters: 基础通道，论文轻量结构默认建议 16
        drop_prob: DropBlock 丢弃概率
        block_size: DropBlock 块大小
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=16, drop_prob=0.1, block_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.drop_prob = drop_prob
        self.block_size = block_size

        self.enc1 = EncoderBlock(in_channels, base_filters, drop_prob=drop_prob, block_size=block_size)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, drop_prob=drop_prob, block_size=block_size)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, drop_prob=drop_prob, block_size=block_size)

        self.bridge_conv1 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False)
        self.bridge_bn1 = nn.BatchNorm2d(base_filters * 8)
        self.bridge_attn = SpatialAttention()
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
    model = SAUNet(in_channels=3, out_channels=1, base_filters=16).to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))

    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
