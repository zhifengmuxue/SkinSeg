"""
Shape Attentive U-Net family blocks.

参考:
1. SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation
   https://arxiv.org/abs/2001.07645
2. CBAM_SAUNet: A novel attention U-Net for effective segmentation of corner cases
   https://doi.org/10.1109/EMBC53108.2024.10782335
3. CA-SAUNet: Coordinate-Attention Enhanced Shape Attentive UNet for medical and agricultural image segmentation
   https://doi.org/10.1007/s10044-025-01590-y

说明:
- 这里抽取 Shape Attentive U-Net 系列中共用的 encoder、dual-attention decoder
  与 gated shape stream，便于 CBAM-SAUNet / CA-SAUNet 共用。
- 当前实现保持你项目里的轻量 U-Net 风格接口，但结构上对齐到
  Shape Attentive U-Net 这条论文路线，而不是之前的 retinal SA-UNet。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover - 仅在未安装 opencv 时兜底
    cv2 = None


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop_prob=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, drop_prob=drop_prob),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, drop_prob=drop_prob),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class GatedSpatialConv2d(nn.Module):
    """
    Shape stream 中的 gated convolution.

    来自 Shape Attentive U-Net 的核心思想:
    使用 1 通道引导图对 shape feature 做空间门控，强调边界相关区域。
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.BatchNorm2d(channels + 1),
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, shape_feat, guidance):
        alpha = self.gate(torch.cat([shape_feat, guidance], dim=1))
        shape_feat = shape_feat * (alpha + 1.0)
        shape_feat = self.proj(shape_feat)
        return shape_feat, alpha


class SpatialAttentionMap(nn.Module):
    """
    Shape Attentive U-Net dual-attention decoder 中的空间注意力分支。
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.down = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.phi = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.relu(self.bn(self.down(x)))
        return torch.sigmoid(self.phi(x))


class SEAttention(nn.Module):
    """
    原始 Shape Attentive U-Net decoder 使用的 SE 通道注意力。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.relu(self.fc1(attn))
        attn = self.sigmoid(self.fc2(attn))
        return x * attn


class CBAMChannelAttention(nn.Module):
    """
    CBAM 的通道注意力分支。

    CBAM_SAUNet 论文摘要说明其改动重点在 decoder block 中引入
    基于 CBAM channel attention 的变体，因此这里仅替换 channel path，
    保留 SAUNet 的空间注意力分支。
    """

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


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention.

    CA-SAUNet 论文说明是用 CA 替换原 SAUNet decoder 中的 SE 模块，
    因此这里作为 dual-attention decoder 的 channel path 使用。
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class DualAttentionDecoderBlock(nn.Module):
    """
    Shape Attentive U-Net 的 dual-attention decoder block.

    F(X) = C(X) * (1 + S(X))
    其中:
    - S(X): spatial attention path
    - C(X): channel attention path
    """

    def __init__(self, in_channels, skip_channels, out_channels, channel_attention: nn.Module, drop_prob=0.0):
        super().__init__()
        self.channel_attention = channel_attention
        self.fuse = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, drop_prob=drop_prob),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, drop_prob=drop_prob),
        )
        self.spatial_attention = SpatialAttentionMap(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([x, skip], dim=1))
        spatial = self.spatial_attention(fused)
        channel = self.channel_attention(fused)
        out = channel * (1.0 + spatial.expand_as(channel))
        return out, spatial


class ShapeStream(nn.Module):
    """
    Shape Attentive U-Net 的 gated shape stream.
    """

    def __init__(self, image_channels, base_filters):
        super().__init__()
        self.shape_proj = nn.Conv2d(base_filters, base_filters, kernel_size=1, bias=False)
        self.shape_res1 = ResidualBlock(base_filters)
        self.shape_res2 = ResidualBlock(base_filters)
        self.shape_res3 = ResidualBlock(base_filters)

        self.gate1 = GatedSpatialConv2d(base_filters)
        self.gate2 = GatedSpatialConv2d(base_filters)
        self.gate3 = GatedSpatialConv2d(base_filters)

        self.guide2 = nn.Conv2d(base_filters * 2, 1, kernel_size=1, bias=False)
        self.guide3 = nn.Conv2d(base_filters * 4, 1, kernel_size=1, bias=False)
        self.guide4 = nn.Conv2d(base_filters * 8, 1, kernel_size=1, bias=False)

        self.edge_head = nn.Conv2d(base_filters, 1, kernel_size=1, bias=True)
        self.edge_fuse = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.edge_expand = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.image_channels = image_channels

    def _image_edges(self, x):
        gray = x.detach().mean(dim=1, keepdim=True)
        gray = gray - gray.amin(dim=(2, 3), keepdim=True)
        gray = gray / (gray.amax(dim=(2, 3), keepdim=True) + 1e-6)

        if cv2 is None:
            dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            dx = F.pad(dx, (0, 1, 0, 0))
            dy = F.pad(dy, (0, 0, 0, 1))
            return (dx + dy).clamp(0, 1)

        gray_np = (gray.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
        canny = np.zeros((gray_np.shape[0], 1, gray_np.shape[1], gray_np.shape[2]), dtype=np.float32)
        for idx in range(gray_np.shape[0]):
            canny[idx, 0] = cv2.Canny(gray_np[idx], 10, 100).astype(np.float32) / 255.0
        return torch.from_numpy(canny).to(x.device)

    def forward(self, image, e1, e2, e3, e4):
        shape = self.shape_res1(self.shape_proj(e1))

        g2 = F.interpolate(self.guide2(e2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate1(shape, g2)
        shape = self.shape_res2(shape)

        g3 = F.interpolate(self.guide3(e3), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate2(shape, g3)
        shape = self.shape_res3(shape)

        g4 = F.interpolate(self.guide4(e4), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate3(shape, g4)

        edge_pred = torch.sigmoid(self.edge_head(shape))
        image_edge = self._image_edges(image)
        fused_edge = torch.sigmoid(self.edge_fuse(torch.cat([edge_pred, image_edge], dim=1)))
        return self.edge_expand(fused_edge)


class ShapeAttentiveUNetBase(nn.Module):
    """
    面向 CBAM-SAUNet / CA-SAUNet 的共用 Shape Attentive U-Net 基座。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        base_filters,
        drop_prob,
        channel_attention_factory: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = EncoderBlock(in_channels, base_filters, drop_prob=drop_prob)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, drop_prob=drop_prob)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, drop_prob=drop_prob)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8, drop_prob=drop_prob)

        self.center = nn.Sequential(
            ConvBNReLU(base_filters * 8, base_filters * 16, kernel_size=3, drop_prob=drop_prob),
            ConvBNReLU(base_filters * 16, base_filters * 16, kernel_size=3, drop_prob=drop_prob),
        )

        self.dec4 = DualAttentionDecoderBlock(
            base_filters * 16,
            base_filters * 8,
            base_filters * 8,
            channel_attention=channel_attention_factory(base_filters * 8),
            drop_prob=drop_prob,
        )
        self.dec3 = DualAttentionDecoderBlock(
            base_filters * 8,
            base_filters * 4,
            base_filters * 4,
            channel_attention=channel_attention_factory(base_filters * 4),
            drop_prob=drop_prob,
        )
        self.dec2 = DualAttentionDecoderBlock(
            base_filters * 4,
            base_filters * 2,
            base_filters * 2,
            channel_attention=channel_attention_factory(base_filters * 2),
            drop_prob=drop_prob,
        )
        self.dec1 = DualAttentionDecoderBlock(
            base_filters * 2,
            base_filters,
            base_filters,
            channel_attention=channel_attention_factory(base_filters),
            drop_prob=drop_prob,
        )

        self.shape_stream = ShapeStream(image_channels=in_channels, base_filters=base_filters)
        self.final_fuse = nn.Sequential(
            ConvBNReLU(base_filters * 2, base_filters, kernel_size=3, drop_prob=drop_prob),
            nn.Conv2d(base_filters, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))

        d4, _ = self.dec4(center, e4)
        d3, _ = self.dec3(d4, e3)
        d2, _ = self.dec2(d3, e2)
        d1, _ = self.dec1(d2, e1)

        edge_feat = self.shape_stream(x, e1, e2, e3, e4)
        logits = self.final_fuse(torch.cat([d1, edge_feat], dim=1))
        return logits
