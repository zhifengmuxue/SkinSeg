"""
模型工厂

统一根据 config 中的 MODEL_TYPE 构建分割模型，
避免 train / eval / predict / visualize 四处分别写 if-else。
"""

from model.attention_unet import AttentionUNet
from model.ca_sa_unet import CASAUNet
from model.cbam_sa_unet import CBAMSAUNet
from model.sa_unet import SAUNet
from model.unet import UNet


def build_model(cfg):
    model_type = cfg.MODEL_TYPE.lower()
    model_kwargs = dict(getattr(cfg, "MODEL_KWARGS", {}) or {})

    if model_type == "unet":
        model_kwargs.setdefault("base_filters", cfg.BASE_FILTERS)
        model_kwargs.setdefault("bilinear", cfg.BILINEAR)
        model = UNet(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            **model_kwargs,
        )
    elif model_type == "attention_unet":
        model_kwargs.setdefault("base_filters", cfg.BASE_FILTERS)
        model_kwargs.setdefault("bilinear", cfg.BILINEAR)
        model = AttentionUNet(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            **model_kwargs,
        )
    elif model_type == "sa_unet":
        model_kwargs.setdefault("base_filters", cfg.BASE_FILTERS)
        model = SAUNet(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            **model_kwargs,
        )
    elif model_type == "cbam_sa_unet":
        model_kwargs.setdefault("base_filters", cfg.BASE_FILTERS)
        model = CBAMSAUNet(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            **model_kwargs,
        )
    elif model_type == "ca_sa_unet":
        model_kwargs.setdefault("base_filters", cfg.BASE_FILTERS)
        model = CASAUNet(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"不支持的模型类型: {cfg.MODEL_TYPE}，可选: "
            f"'unet' / 'attention_unet' / 'sa_unet' / 'cbam_sa_unet' / 'ca_sa_unet'"
        )

    return model.to(cfg.DEVICE)
