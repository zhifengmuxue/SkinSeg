"""
Wavelet-Guided Shape SAUNet (WGS-SAUNet).

设计说明:
- 以 Shape Attentive U-Net (SAUNet) 为主干，保持 dual-attention decoder
  与 segmentation loss 不变。
- 使用固定 Haar DWT 从输入图像提取高频子带，生成 wavelet boundary prior。
- 该 prior 仅注入 shape stream，用于增强边界引导，而不是替换编码器中的
  pooling/upsampling，以降低与常见 Wavelet U-Net 路线的重合度。

参考:
1. Shape Attentive U-Net for Interpretable Medical Image Segmentation
   https://arxiv.org/abs/2001.07645
2. 本实现为基于 SAUNet 的扩展设计，不对应单篇已发表 WGS-SAUNet 论文。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.shape_attentive_blocks import ResidualBlock, SEAttention, ShapeAttentiveUNetBase
from model.unet import count_params_flops


class HaarDWT2D(nn.Module):
    """Fixed 2D Haar DWT implemented with tensor slicing."""

    def forward(self, x):
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class WaveletPriorExtractor(nn.Module):
    """Encodes DWT high-frequency bands into a single boundary prior map."""

    def __init__(self, in_channels, hidden_channels=16, use_abs_high=True):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.use_abs_high = use_abs_high
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        _, lh, hl, hh = self.dwt(x)
        if self.use_abs_high:
            lh = lh.abs()
            hl = hl.abs()
            hh = hh.abs()
        prior = self.encoder(torch.cat([lh, hl, hh], dim=1))
        return torch.sigmoid(prior)


class WaveletGuidedGatedSpatialConv2d(nn.Module):
    """Shape-stream gate that additionally conditions on a wavelet prior map."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.BatchNorm2d(channels + 2),
            nn.Conv2d(channels + 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, shape_feat, guidance, wavelet_prior):
        alpha = self.gate(torch.cat([shape_feat, guidance, wavelet_prior], dim=1))
        shape_feat = self.proj(shape_feat * (alpha + 1.0))
        return shape_feat, alpha


class WaveletGuidedShapeStream(nn.Module):
    """SAUNet shape stream augmented with a DWT-derived boundary prior."""

    def __init__(
        self,
        image_channels,
        base_filters,
        wavelet_prior_channels=16,
        wavelet_use_abs_high=True,
        wavelet_fuse_into_edge=True,
    ):
        super().__init__()
        self.prior_extractor = WaveletPriorExtractor(
            in_channels=image_channels,
            hidden_channels=wavelet_prior_channels,
            use_abs_high=wavelet_use_abs_high,
        )
        self.shape_proj = nn.Conv2d(base_filters, base_filters, kernel_size=1, bias=False)
        self.shape_res1 = ResidualBlock(base_filters)
        self.shape_res2 = ResidualBlock(base_filters)
        self.shape_res3 = ResidualBlock(base_filters)

        self.gate1 = WaveletGuidedGatedSpatialConv2d(base_filters)
        self.gate2 = WaveletGuidedGatedSpatialConv2d(base_filters)
        self.gate3 = WaveletGuidedGatedSpatialConv2d(base_filters)

        self.guide2 = nn.Conv2d(base_filters * 2, 1, kernel_size=1, bias=False)
        self.guide3 = nn.Conv2d(base_filters * 4, 1, kernel_size=1, bias=False)
        self.guide4 = nn.Conv2d(base_filters * 8, 1, kernel_size=1, bias=False)

        edge_fuse_in_channels = 3 if wavelet_fuse_into_edge else 2
        self.wavelet_fuse_into_edge = wavelet_fuse_into_edge
        self.edge_head = nn.Conv2d(base_filters, 1, kernel_size=1, bias=True)
        self.edge_fuse = nn.Conv2d(edge_fuse_in_channels, 1, kernel_size=1, bias=False)
        self.edge_expand = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _image_edges(x):
        gray = x.detach().mean(dim=1, keepdim=True)
        gray = gray - gray.amin(dim=(2, 3), keepdim=True)
        gray = gray / (gray.amax(dim=(2, 3), keepdim=True) + 1e-6)

        dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return (dx + dy).clamp(0, 1)

    def forward(self, image, e1, e2, e3, e4):
        prior = F.interpolate(
            self.prior_extractor(image),
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        shape = self.shape_res1(self.shape_proj(e1))

        g2 = F.interpolate(self.guide2(e2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate1(shape, g2, prior)
        shape = self.shape_res2(shape)

        g3 = F.interpolate(self.guide3(e3), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate2(shape, g3, prior)
        shape = self.shape_res3(shape)

        g4 = F.interpolate(self.guide4(e4), size=e1.shape[2:], mode="bilinear", align_corners=False)
        shape, _ = self.gate3(shape, g4, prior)

        edge_pred = torch.sigmoid(self.edge_head(shape))
        image_edge = self._image_edges(image)
        if self.wavelet_fuse_into_edge:
            fused_inputs = torch.cat([edge_pred, image_edge, prior], dim=1)
        else:
            fused_inputs = torch.cat([edge_pred, image_edge], dim=1)
        fused_edge = torch.sigmoid(self.edge_fuse(fused_inputs))
        return self.edge_expand(fused_edge)


class WGSSAUNet(ShapeAttentiveUNetBase):
    """
    DWT-based Wavelet-Guided Shape SAUNet.

    在标准 SAUNet 上，将固定 Haar DWT 得到的高频先验注入 gated shape stream，
    保持 dual-attention decoder 与分割 head 不变，用于公平比较结构收益。
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_filters=64,
        drop_prob=0.1,
        block_size=7,
        wavelet_prior_channels=16,
        wavelet_use_abs_high=True,
        wavelet_fuse_into_edge=True,
    ):
        self.block_size = block_size  # 兼容现有配置接口，当前版本未使用
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            drop_prob=drop_prob,
            channel_attention_factory=lambda channels: SEAttention(channels, reduction=16),
        )
        self.shape_stream = WaveletGuidedShapeStream(
            image_channels=in_channels,
            base_filters=base_filters,
            wavelet_prior_channels=wavelet_prior_channels,
            wavelet_use_abs_high=wavelet_use_abs_high,
            wavelet_fuse_into_edge=wavelet_fuse_into_edge,
        )
        self._init_weights()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WGSSAUNet(in_channels=3, out_channels=1, base_filters=64).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x)
    print("Output shape:", y.shape)
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params: {params_m:.2f} M | FLOPs: {flops_g:.2f} GFLOPs")
