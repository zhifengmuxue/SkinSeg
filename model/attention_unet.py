"""
Attention U-Net

基于论文:
"Attention U-Net: Learning Where to Look for the Pancreas"
https://arxiv.org/abs/1804.03999

实现要点:
1. 保留标准 U-Net 的 encoder-decoder 结构
2. 在每条 skip connection 上加入 additive attention gate
3. 输出保持为 logits，便于与 BCEWithLogitsLoss 等损失函数兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet import DoubleConv, Down, OutConv, count_params_flops


class AttentionGate(nn.Module):
    """Oktay et al. 的 additive attention gate"""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        gate_feat = self.gate_proj(gate)
        skip_feat = self.skip_proj(skip)

        if gate_feat.shape[-2:] != skip_feat.shape[-2:]:
            gate_feat = F.interpolate(gate_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=True)

        attn = self.relu(gate_feat + skip_feat)
        attn = self.psi(attn)
        return skip * attn


class AttUp(nn.Module):
    """上采样 + Attention Gate + concat + DoubleConv"""

    def __init__(self, decoder_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = decoder_channels
        else:
            self.up = nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)
            up_channels = decoder_channels // 2

        self.attn = AttentionGate(
            gate_channels=up_channels,
            skip_channels=skip_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv(up_channels + skip_channels, out_channels, mid_ch=(up_channels + skip_channels) // 2)

    def forward(self, x_decoder, x_skip):
        x_decoder = self.up(x_decoder)

        diff_h = x_skip.size(2) - x_decoder.size(2)
        diff_w = x_skip.size(3) - x_decoder.size(3)
        x_decoder = F.pad(
            x_decoder,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )

        x_skip = self.attn(x_skip, x_decoder)
        x = torch.cat([x_skip, x_decoder], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    2D Attention U-Net

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        base_filters: 基础通道数
        bilinear: 是否使用双线性插值上采样
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)

        self.up1 = AttUp(base_filters * 16 // factor, base_filters * 8, base_filters * 8 // factor, bilinear)
        self.up2 = AttUp(base_filters * 8 // factor, base_filters * 4, base_filters * 4 // factor, bilinear)
        self.up3 = AttUp(base_filters * 4 // factor, base_filters * 2, base_filters * 2 // factor, bilinear)
        self.up4 = AttUp(base_filters * 2 // factor, base_filters, base_filters, bilinear)
        self.outc = OutConv(base_filters, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1, base_filters=64).to(device)
    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=("input_size", "output_size", "num_params"))

    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
