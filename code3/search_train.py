import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import sys
import os
import time
import random
import numpy as np

# 将 code2 和 code3 加入路径
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.join(os.path.dirname(CODE_DIR), 'code2')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CODE2_DIR)

from search_config import (
    suggest_params,
    INPUT_LENGTH, NUM_TYPE_CLASSES, NUM_SIZE_CLASSES,
    N_EPOCHS_PER_TRIAL, EARLY_STOP_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    ADD_NOISE_SNR, NORMALIZE, RANDOM_SEED,
)
from search_model import build_model
from dataset import CWRUDataset


def set_seed(seed):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, type_acc, size_acc, device):
    model.train()
    total_loss = 0.0

    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)

        optimizer.zero_grad()
        type_logits, size_logits = model(waveforms)

        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'type_acc': type_acc.compute().item(),
        'size_acc': size_acc.compute().item(),
    }


@torch.no_grad()
def validate(model, loader, criterion, type_acc, size_acc, device):
    """验证集评估。

    返回：
        dict: 包含 loss, type_acc, size_acc, joint_acc
    """
    model.eval()
    total_loss = 0.0
    all_type_preds = []
    all_size_preds = []
    all_type_labels = []
    all_size_labels = []

    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)

        type_logits, size_logits = model(waveforms)
        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        total_loss += loss.item()

        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)

        all_type_preds.append(torch.argmax(type_logits, dim=1))
        all_size_preds.append(torch.argmax(size_logits, dim=1))
        all_type_labels.append(type_labels)
        all_size_labels.append(size_labels)

    n_batches = len(loader)
    type_preds = torch.cat(all_type_preds)
    size_preds = torch.cat(all_size_preds)
    type_lbl = torch.cat(all_type_labels)
    size_lbl = torch.cat(all_size_labels)

    joint_correct = ((type_preds == type_lbl) & (size_preds == size_lbl)).sum().item()
    joint_acc = joint_correct / len(type_lbl)

    return {
        'loss': total_loss / n_batches,
        'type_acc': type_acc.compute().item(),
        'size_acc': size_acc.compute().item(),
        'joint_acc': joint_acc,
    }


def objective(trial):
    """Optuna 多目标优化目标函数。

    每个 trial 执行完整的训练+验证流程，返回两个目标值：
        val_type_acc, val_size_acc

    Optuna 会同时最大化这两个目标，探索 Pareto 前沿。
    """
    # 休息
    mins = 0
    if mins:
        print(f"暂停{mins}分钟")
        time.sleep(mins*60)

    # ---- 采样超参数 ----
    params = suggest_params(trial)

    # ---- 设置设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 固定种子 ----
    set_seed(RANDOM_SEED)

    # ---- 加载数据 ----
    train_ds = CWRUDataset('train', add_noise_snr=ADD_NOISE_SNR, normalize=NORMALIZE)
    val_ds = CWRUDataset('val', normalize=NORMALIZE)

    train_loader = DataLoader(
        train_ds, batch_size=params['batch_size'], shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # ---- 构建模型 ----
    try:
        model = build_model(params, input_length=INPUT_LENGTH,
                            num_type_classes=NUM_TYPE_CLASSES,
                            num_size_classes=NUM_SIZE_CLASSES).to(device)
    except Exception as e:
        return 0, 0
    # 记录参数量到 trial
    n_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr('n_params', n_params)

    # ---- 优化器和调度器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE
    )

    # ---- 训练循环 ----
    best_val_type = 0.0
    best_val_size = 0.0
    best_val_joint = 0.0
    patience_counter = 0

    for epoch in range(1, N_EPOCHS_PER_TRIAL + 1):
        # 训练
        train_type_acc = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(device)
        train_size_acc = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(device)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            train_type_acc, train_size_acc, device
        )

        # 验证
        val_type_acc = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(device)
        val_size_acc = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(device)
        val_metrics = validate(
            model, val_loader, criterion,
            val_type_acc, val_size_acc, device
        )

        # 学习率调度
        scheduler.step(val_metrics['loss'])

        # 记录中间值（供 Optuna 剪枝和可视化）
        #trial.report(val_metrics['type_acc'], epoch)
        #trial.report(val_metrics['size_acc'], epoch)

        # 早停检查：以 joint_acc 为参考
        if val_metrics['joint_acc'] > best_val_joint:
            best_val_joint = val_metrics['joint_acc']
            best_val_type = val_metrics['type_acc']
            best_val_size = val_metrics['size_acc']
            patience_counter = 0
        else:
            patience_counter += 1

        # Optuna 剪枝：如果当前 epoch 表现远差于历史最佳，提前终止
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    # 存储最佳验证指标
    trial.set_user_attr('best_val_joint', best_val_joint)
    trial.set_user_attr('best_val_type', best_val_type)
    trial.set_user_attr('best_val_size', best_val_size)

    return best_val_type, best_val_size


if __name__ == '__main__':
    pass