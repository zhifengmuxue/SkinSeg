import os
import yaml
import torch

class Config:
    def __init__(self, mode="train"):
        """
        统一配置加载器
        mode: "train" | "eval" | "predict" | "visualize"
        """
        yaml_name = os.environ.get("SEGMENT_CONFIG", "config.yml")
        if os.path.isabs(yaml_name):
            yaml_path = yaml_name
        else:
            yaml_path = os.path.join(os.path.dirname(__file__), yaml_name)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 1. 加载全局配置
        self.DEVICE_STR   = cfg.get("device", "cuda")
        self.DEVICE       = torch.device(self.DEVICE_STR) if torch.cuda.is_available() and self.DEVICE_STR == 'cuda' else torch.device('cpu')
        self.SEED         = cfg.get("seed", 42)
        self.NUM_WORKERS  = cfg.get("num_workers", 4)
        self.PIN_MEMORY   = cfg.get("pin_memory", True)
        self.DATA_DIR     = cfg.get("data_dir", "d:/code/segment/dataset")
        self.IMG_SIZE     = cfg.get("img_size", 256)

        # 2. 模型配置
        self.MODEL_TYPE   = cfg.get("model_type", "unet")
        self.IN_CHANNELS  = cfg.get("in_channels", 3)
        self.OUT_CHANNELS = cfg.get("out_channels", 1)
        self.BASE_FILTERS = cfg.get("base_filters", 64)
        self.BILINEAR     = cfg.get("bilinear", True)
        self.THRESHOLD    = cfg.get("threshold", 0.5)
        self.MODEL_KWARGS = cfg.get("model_kwargs", {})

        # 3. 加载特定模式的配置
        if mode in cfg:
            for k, v in cfg[mode].items():
                setattr(self, k.upper(), v)
