"""
分割任务损失函数
"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """BCEWithLogits + Dice 联合损失 (兼容 AMP)"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def compute_components(self, pred, target):
        # pred:  (B, 1, H, W)  logits
        # target: (B, 1, H, W)  0/1
        bce = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return total, bce, dice_loss

    def forward(self, pred, target):
        total, _, _ = self.compute_components(pred, target)
        return total
