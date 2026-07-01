"""
U-Net 语义分割模型

基于论文: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
https://arxiv.org/abs/1505.04597

架构: 对称 Encoder-Decoder + Skip Connection
输入:  (B, 3, H, W)  皮肤镜图像
输出:  (B, 1, H, W)  二值分割 mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 基础模块
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样: MaxPool → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样: Upsample → Conv → concat skip → DoubleConv"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 上采样后通道不变，拼接时 in_ch = up_ch + skip_ch
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            # 转置卷积上采样，通道减半
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: 待上采样的特征 (decoder / bottleneck)
        # x2: 对应 encoder 层的 skip connection
        x1 = self.up(x1)
        # 处理尺寸不匹配 (输入 H/W 不是 16 的倍数时)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层: 1×1 Conv → 1 通道"""

    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------------------
# U-Net 主模型
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """
    经典 U-Net 分割网络

    Args:
        in_channels  : 输入通道数 (3 for RGB)
        out_channels : 输出通道数 (1 for binary mask)
        base_filters : 第一层通道数 (默认 64)
        bilinear     : 是否使用双线性插值上采样 (False 则用转置卷积)
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # ---- Encoder ----
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)

        # ---- Decoder ----
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)

        # ---- Output ----
        self.outc = OutConv(base_filters, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # base_filters
        x2 = self.down1(x1)    # base_filters * 2
        x3 = self.down2(x2)    # base_filters * 4
        x4 = self.down3(x3)    # base_filters * 8
        x5 = self.down4(x4)    # bottleneck: base_filters * 16 // factor

        # Decoder (with skip connections)
        x = self.up1(x5, x4)   # base_filters * 8 // factor
        x = self.up2(x, x3)    # base_filters * 4 // factor
        x = self.up3(x, x2)    # base_filters * 2 // factor
        x = self.up4(x, x1)    # base_filters

        return self.outc(x)  # (B, out_channels, H, W)  logits


# -----------------------------------------------------------------------------
# 模型统计工具
# -----------------------------------------------------------------------------
def count_params_flops(model, input_shape=(3, 256, 256), device=None):
    """
    使用 torchinfo 计算 Params 与 FLOPs

    Args:
        model:        nn.Module
        input_shape:  (C, H, W)
        device:       torch.device, 如果为 None 则不移动模型

    Returns:
        params_m: float (百万)
        flops_g:  float (GFLOPs)
    """
    from torchinfo import summary

    model.eval()
    stat = summary(model, input_size=(1, *input_shape), verbose=0, device=device)
    params_m = stat.total_params / 1e6
    flops_g = stat.total_mult_adds / 1e9
    return params_m, flops_g


# -----------------------------------------------------------------------------
# 测试入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_filters=64).to(device)

    summary(model, input_size=(2, 3, 256, 256), depth=5, col_names=(
        "input_size", "output_size", "num_params", "kernel_size", "mult_adds",
    ))

    # 验证 forward
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    params = sum(p.numel() for p in model.parameters())
    params_m, flops_g = count_params_flops(model, input_shape=(3, 256, 256), device=device)
    print(f"\n参数量: {params:,}")
    print(f"Params : {params_m:.2f} M")
    print(f"FLOPs  : {flops_g:.2f} GFLOPs")
    print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Probs  range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
