import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import numpy as np
import os
import sys
import json
import time
import random
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径设置
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.join(os.path.dirname(CODE_DIR), 'code2')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CODE2_DIR)

from search_config import (
    INPUT_LENGTH, NUM_TYPE_CLASSES, NUM_SIZE_CLASSES,
    ADD_NOISE_SNR, NORMALIZE, RANDOM_SEED,
)
from search_model import build_model
from dataset import CWRUDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUTS_DIR = os.path.join(CODE_DIR, 'outputs')
BEST_PARAMS_PATH = os.path.join(OUTPUTS_DIR, 'best_params.json')

RETRAIN_EPOCHS = 160
RETRAIN_PATIENCE = 30
RETRAIN_LR_PATIENCE = 15
RETRAIN_LR_FACTOR = 0.5

TYPE_NAMES = ['Normal', 'Inner', 'Outer', 'Ball']
SIZE_NAMES = ['None', '0.007in', '0.014in', '0.021in']


def set_seed(seed):
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
    n = len(loader)
    return {'loss': total_loss / n, 'type_acc': type_acc.compute().item(), 'size_acc': size_acc.compute().item()}


@torch.no_grad()
def evaluate(model, loader, criterion, type_acc, size_acc, device):
    model.eval()
    total_loss = 0.0
    all_tp, all_sp, all_tl, all_sl = [], [], [], []
    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)
        type_logits, size_logits = model(waveforms)
        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        total_loss += loss.item()
        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)
        all_tp.append(torch.argmax(type_logits, dim=1))
        all_sp.append(torch.argmax(size_logits, dim=1))
        all_tl.append(type_labels)
        all_sl.append(size_labels)
    n = len(loader)
    tp = torch.cat(all_tp)
    sp = torch.cat(all_sp)
    tl = torch.cat(all_tl)
    sl = torch.cat(all_sl)
    joint = (type_acc.compute().item() + size_acc.compute().item())/2.0
    return {
        'loss': total_loss / n,
        'type_acc': type_acc.compute().item(),
        'size_acc': size_acc.compute().item(),
        'joint_acc': joint,
        'type_preds': tp.cpu().numpy(),
        'size_preds': sp.cpu().numpy(),
        'type_labels': tl.cpu().numpy(),
        'size_labels': sl.cpu().numpy(),
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=12)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    #plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Confusion matrix saved: ' + save_path)


def main():
    print('=' * 70)
    print('CWRU Bearing Fault Diagnosis -- Retrain with Best Params (code3)')
    print('Device: ' + str(DEVICE))
    print('Start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 70)

    # ---- 加载最佳参数 ----
    if not os.path.exists(BEST_PARAMS_PATH):
        print('ERROR: best_params.json not found at ' + BEST_PARAMS_PATH)
        print('Please run run_search.py first.')
        return

    with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
        best_result = json.load(f)

    best_params = best_result['params']
    print()
    print('Loaded best params from search:')
    print('  val_type_acc:  ' + str(best_result['val_type_acc']))
    print('  val_size_acc:  ' + str(best_result['val_size_acc']))
    print('  val_joint_acc: ' + str(best_result['val_joint_acc']))
    print('  n_params:      ' + str(best_result['n_params']))

    set_seed(RANDOM_SEED)

    # ---- 加载数据 ----
    print()
    print('[1/4] Loading data...')
    train_ds = CWRUDataset('train', add_noise_snr=ADD_NOISE_SNR, normalize=NORMALIZE)
    val_ds = CWRUDataset('val', normalize=NORMALIZE)
    test_ds = CWRUDataset('test', normalize=NORMALIZE)

    batch_size = best_params.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    print('  Train: ' + str(len(train_ds)) + '  Val: ' + str(len(val_ds)) + '  Test: ' + str(len(test_ds)))

    # ---- 补全 channels 参数 ----
    # Optuna 保存的是 base_channels + channel_growth，需要还原为 channels 列表
    if 'channels' not in best_params:
        base_channels = best_params.get('base_channels', 16)
        channel_growth = best_params.get('channel_growth', 1.5)
        num_layers = best_params.get('num_conv_layers', 4)
        channels = []
        for i in range(num_layers):
            ch = int(base_channels * (channel_growth ** i))
            ch = min(ch, 512)
            channels.append(ch)
        best_params['channels'] = channels

    # ---- 构建模型 ----
    print()
    print('[2/4] Building model with best params...')
    model = build_model(best_params, input_length=INPUT_LENGTH,
                        num_type_classes=NUM_TYPE_CLASSES,
                        num_size_classes=NUM_SIZE_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print('  Parameters: ' + str(n_params))

    criterion = nn.CrossEntropyLoss()

    ckpt = os.path.exists("code3/outputs/best_model_retrained.pth")
    if ckpt:
        best_model_path = "code3/outputs/best_model_retrained.pth"
        print("使用存在的权重！")
    else:
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                                weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=RETRAIN_LR_FACTOR, patience=RETRAIN_LR_PATIENCE)

        # ---- 训练 ----
        print()
        print('[3/4] Retraining (max ' + str(RETRAIN_EPOCHS) + ' epochs)...')
        print('-' * 70)

        best_val_joint = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(OUTPUTS_DIR, 'best_model_retrained.pth')

        for epoch in range(1, RETRAIN_EPOCHS + 1):
            t0 = time.time()

            tta = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(DEVICE)
            tsa = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(DEVICE)
            train_m = train_one_epoch(model, train_loader, criterion, optimizer, tta, tsa, DEVICE)

            vta = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(DEVICE)
            vsa = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(DEVICE)
            val_m = evaluate(model, val_loader, criterion, vta, vsa, DEVICE)

            scheduler.step(val_m['loss'])
            dt = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                print('Epoch {:3d} ({:.1f}s)  Train Loss={:.4f} T={:.4f} S={:.4f}  Val Loss={:.4f} T={:.4f} S={:.4f} J={:.4f}'.format(
                    epoch, dt, train_m['loss'], train_m['type_acc'], train_m['size_acc'],
                    val_m['loss'], val_m['type_acc'], val_m['size_acc'], val_m['joint_acc']))

            if val_m['joint_acc'] > best_val_joint:
                best_val_joint = val_m['joint_acc']
                best_epoch = epoch
                patience_counter = 0
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': {k: v for k, v in val_m.items() if k not in ('type_preds', 'size_preds', 'type_labels', 'size_labels')},
                    'params': best_params,
                }, best_model_path)
                print('  >>> Saved (Val Joint={:.4f})'.format(best_val_joint))
            else:
                patience_counter += 1

            if patience_counter >= RETRAIN_PATIENCE:
                print('Early stopping at epoch ' + str(epoch))
                break

        print()
        print('Best epoch: ' + str(best_epoch) + '  Best val joint: {:.4f}'.format(best_val_joint))

    # ---- 加载最佳模型并测试 ----
    print()
    print('[4/4] Evaluating on test set...')
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    tta = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(DEVICE)
    tsa = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(DEVICE)
    test_m = evaluate(model, test_loader, criterion, tta, tsa, DEVICE)

    type_acc = test_m['type_acc']
    size_acc = test_m['size_acc']
    joint_acc = test_m['joint_acc']

    # torchmetrics F1
    tp_t = torch.tensor(test_m['type_preds'])
    sp_t = torch.tensor(test_m['size_preds'])
    tl_t = torch.tensor(test_m['type_labels'])
    sl_t = torch.tensor(test_m['size_labels'])
    type_f1 = F1Score(task='multiclass', num_classes=NUM_TYPE_CLASSES, average='macro')(tp_t, tl_t).item()
    size_f1 = F1Score(task='multiclass', num_classes=NUM_SIZE_CLASSES, average='macro')(sp_t, sl_t).item()

    print()
    print('=' * 70)
    print('  TEST RESULTS')
    print('=' * 70)
    print('  Type Accuracy:   {:.4f}'.format(type_acc))
    print('  Type Macro F1:   {:.4f}'.format(type_f1))
    print('  Size Accuracy:   {:.4f}'.format(size_acc))
    print('  Size Macro F1:   {:.4f}'.format(size_f1))
    print('  Joint Accuracy:  {:.4f}'.format(joint_acc))

    print()
    print('=== Fault Type Classification Report ===')
    print(classification_report(test_m['type_labels'], test_m['type_preds'],
                                target_names=TYPE_NAMES, digits=4))

    print('=== Fault Size Classification Report ===')
    print(classification_report(test_m['size_labels'], test_m['size_preds'],
                                target_names=SIZE_NAMES, digits=4))

    # ---- 混淆矩阵 ----
    print()
    print('=== Generating Confusion Matrices ===')
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    cm_type = confusion_matrix(test_m['type_labels'], test_m['type_preds'])
    plot_confusion_matrix(cm_type, TYPE_NAMES, 'Fault Type Confusion Matrix (Optuna Best)',
                            os.path.join(OUTPUTS_DIR, 'confusion_matrix_type_optuna.png'))

    cm_size = confusion_matrix(test_m['size_labels'], test_m['size_preds'])
    plot_confusion_matrix(cm_size, SIZE_NAMES, 'Fault Size Confusion Matrix (Optuna Best)',
                            os.path.join(OUTPUTS_DIR, 'confusion_matrix_size_optuna.png'))

if __name__ == '__main__':
    main()
