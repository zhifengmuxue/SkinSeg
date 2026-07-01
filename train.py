"""
U-Net 皮肤病变分割训练脚本

基于 ISIC 2018 Task 1 数据集进行二值分割训练。
损失函数: BCE + Dice Loss，评估指标: IoU / Dice。
"""

import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data.dataset import get_dataloaders
from losses import BCEDiceLoss
from model.factory import build_model


# ==============================================================================
# 评估指标
# ==============================================================================
def compute_metrics(pred, target, threshold=0.5):
    """计算 IoU 和 Dice (pred 为 logits)"""
    pred_sigmoid = torch.sigmoid(pred)
    pred_bin = (pred_sigmoid > threshold).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()


# ==============================================================================
# 工具函数
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def config_to_markdown(cfg):
    lines = []
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def denormalize_images(images):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def make_overlay_tensor(images, masks):
    overlay = images.clone()
    mask_bool = masks > 0.5
    overlay[:, 0:1] = torch.where(mask_bool, overlay[:, 0:1] * 0.5 + 0.5, overlay[:, 0:1])
    overlay[:, 1:2] = torch.where(mask_bool, overlay[:, 1:2] * 0.5, overlay[:, 1:2])
    overlay[:, 2:3] = torch.where(mask_bool, overlay[:, 2:3] * 0.5, overlay[:, 2:3])
    return overlay


def log_tensorboard_epoch(writer, epoch,
                          train_loss, train_bce, train_dice_loss, train_iou, train_dice,
                          val_loss, val_bce, val_dice_loss, val_iou, val_dice,
                          current_lr, best_dice):
    writer.add_scalar("loss/total_train", train_loss, epoch)
    writer.add_scalar("loss/total_val", val_loss, epoch)
    writer.add_scalar("loss/bce_train", train_bce, epoch)
    writer.add_scalar("loss/bce_val", val_bce, epoch)
    writer.add_scalar("loss/dice_train", train_dice_loss, epoch)
    writer.add_scalar("loss/dice_val", val_dice_loss, epoch)
    writer.add_scalar("metrics/iou_train", train_iou, epoch)
    writer.add_scalar("metrics/iou_val", val_iou, epoch)
    writer.add_scalar("metrics/dice_train", train_dice, epoch)
    writer.add_scalar("metrics/dice_val", val_dice, epoch)
    writer.add_scalar("metrics/best_dice", best_dice, epoch)
    writer.add_scalar("lr/current", current_lr, epoch)


def log_tensorboard_visuals(writer, epoch, images, masks, logits, max_images=4):
    images = images[:max_images].detach().cpu()
    masks = masks[:max_images].detach().cpu()
    probs = torch.sigmoid(logits[:max_images].detach().cpu())
    preds = (probs > 0.5).float()

    images = denormalize_images(images)
    gt_overlay = make_overlay_tensor(images, masks)
    pred_overlay = make_overlay_tensor(images, preds)

    vis = torch.cat([images, gt_overlay, pred_overlay], dim=3)
    writer.add_images("visuals/val_image_gt_overlay_pred_overlay", vis, epoch)


# ==============================================================================
# 训练 / 验证
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        optimizer.zero_grad()

        if cfg.USE_AMP:
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss, bce_loss, dice_loss = criterion.compute_components(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss, bce_loss, dice_loss = criterion.compute_components(preds, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice_loss += dice_loss.item()
        iou, dice = compute_metrics(preds.detach(), masks)
        total_iou += iou
        total_dice += dice

        if batch_idx % cfg.LOG_INTERVAL == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{bce_loss.item():.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_bce / n, total_dice_loss / n, total_iou / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    visual_batch = None

    pbar = tqdm(loader, desc="Val", ncols=100, leave=False)
    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        preds = model(imgs)
        loss, bce_loss, dice_loss = criterion.compute_components(preds, masks)

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice_loss += dice_loss.item()
        iou, dice = compute_metrics(preds, masks)
        total_iou += iou
        total_dice += dice

        if batch_idx == 0:
            visual_batch = (
                imgs.detach().cpu(),
                masks.detach().cpu(),
                preds.detach().cpu(),
            )

        pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{bce_loss.item():.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_bce / n, total_dice_loss / n, total_iou / n, total_dice / n, visual_batch


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    cfg = Config(mode="train")
    
    # --- 随机种子 ---
    set_seed(cfg.SEED)

    # --- 输出目录 ---
    run_dir = os.path.join(cfg.OUTPUT_DIR, cfg.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, 'best.pth')
    last_path = os.path.join(run_dir, 'last.pth')

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
    writer.add_text("run/config", config_to_markdown(cfg), 0)
    writer.add_text("run/visual_legend", "每个样例从左到右依次为: 原图 | GT Overlay | Pred Overlay", 0)

    # --- 数据加载 ---
    train_loader, val_loader, _ = get_dataloaders(
        cfg.DATA_DIR, img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
    )

    print(f"训练集: {len(train_loader.dataset)} 张")
    print(f"验证集: {len(val_loader.dataset)} 张")

    # --- 模型 ---
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters())

    # --- 损失 / 优化器 / 调度器 ---
    criterion = BCEDiceLoss(bce_weight=cfg.BCE_WEIGHT, dice_weight=cfg.DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda') if cfg.USE_AMP else None

    if cfg.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.MIN_LR)
    elif cfg.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.STEP_GAMMA)
    elif cfg.LR_SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=cfg.MIN_LR)
    else:
        scheduler = None

    # --- 断点续训 ---
    start_epoch = 1
    best_dice = 0.0

    resume_path = cfg.RESUME
    if not resume_path and os.path.exists(last_path):
        resume_path = last_path  # 自动检测 last.pth

    if resume_path and os.path.exists(resume_path):
        print(f"加载断点: {resume_path}")
        ckpt = torch.load(resume_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        # 调度器状态恢复
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # scaler 状态恢复
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"  续训起点: Epoch {start_epoch} | 历史最佳 Dice: {best_dice:.4f}")

    print(f"{cfg.MODEL_TYPE} 参数量: {params:,}")

    # --- 日志头 ---
    print(f"\n{'='*55}")
    print(f"  设备: {cfg.DEVICE}  |  AMP: {cfg.USE_AMP}")
    print(f"  Batch: {cfg.BATCH_SIZE}  |  LR: {cfg.LEARNING_RATE}  |  Epochs: {cfg.NUM_EPOCHS}")
    print(f"  输出: {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {run_dir}/tensorboard")
    print(f"{'='*55}\n")

    # --- 训练循环 ---
    t_start = time.time()

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_bce, train_dice_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg,
        )

        # 验证
        val_loss, val_bce, val_dice_loss, val_iou, val_dice, val_visual_batch = validate(
            model, val_loader, criterion, cfg
        )

        # 调度器
        if scheduler is not None:
            if cfg.LR_SCHEDULER == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice_loss:.4f}) "
              f"IoU: {train_iou:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, Dice: {val_dice_loss:.4f}) "
              f"IoU: {val_iou:.4f} Dice: {val_dice:.4f} | "
              f"Time: {format_time(epoch_time)}")

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice

        # TensorBoard
        log_tensorboard_epoch(
            writer=writer,
            epoch=epoch,
            train_loss=train_loss,
            train_bce=train_bce,
            train_dice_loss=train_dice_loss,
            train_iou=train_iou,
            train_dice=train_dice,
            val_loss=val_loss,
            val_bce=val_bce,
            val_dice_loss=val_dice_loss,
            val_iou=val_iou,
            val_dice=val_dice,
            current_lr=current_lr,
            best_dice=best_dice,
        )
        if val_visual_batch is not None and (epoch == start_epoch or epoch % cfg.TB_IMAGE_EVERY == 0):
            log_tensorboard_visuals(
                writer=writer,
                epoch=epoch,
                images=val_visual_batch[0],
                masks=val_visual_batch[1],
                logits=val_visual_batch[2],
                max_images=cfg.TB_NUM_IMAGES,
            )
        writer.flush()

        # --- 保存 last.pth (断点续训) ---
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(ckpt, last_path)

        # --- 保存 best.pth ---
        if improved:
            torch.save(ckpt, best_path)
            print(f"  >> 已保存最佳模型 (Dice: {best_dice:.4f})")

    # --- 训练结束 ---
    total_time = time.time() - t_start
    print(f"\n训练完成! 总耗时: {format_time(total_time)} | 最佳 Val Dice: {best_dice:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
