import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import numpy as np
import os, sys, json, time, random
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.join(os.path.dirname(CODE_DIR), 'code2')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CODE2_DIR)

from search_config_2d import (INPUT_LENGTH, NUM_TYPE_CLASSES, NUM_SIZE_CLASSES,
                                ADD_NOISE_SNR, NORMALIZE, RANDOM_SEED,
                                N_FFT, HOP_LENGTH, FREQ_BINS)
from search_model_2d import build_model   
from dataset import CWRUDataset
import stft_transform_2d as stft_transform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUTS_DIR = os.path.join(CODE_DIR, 'outputs')
BEST_PARAMS_PATH = os.path.join(OUTPUTS_DIR, 'best_params_2d.json')

RETRAIN_EPOCHS = 150
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    type_acc = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(device)
    size_acc = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(device)
    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)
        specs = stft_transform.batch_to_spectrograms(waveforms).to(device)
        optimizer.zero_grad()
        type_logits, size_logits = model(specs)
        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)
    n = len(loader)
    return total_loss / n, type_acc.compute().item(), size_acc.compute().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_type_preds, all_size_preds = [], []
    all_type_labels, all_size_labels = [], []
    type_acc = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(device)
    size_acc = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(device)
    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)
        specs = stft_transform.batch_to_spectrograms(waveforms).to(device)
        type_logits, size_logits = model(specs)
        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        total_loss += loss.item()
        type_preds = type_logits.argmax(dim=1)
        size_preds = size_logits.argmax(dim=1)
        all_type_preds.append(type_preds.cpu())
        all_size_preds.append(size_preds.cpu())
        all_type_labels.append(type_labels.cpu())
        all_size_labels.append(size_labels.cpu())
        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)
    n = len(loader)
    all_type_preds = torch.cat(all_type_preds).numpy()
    all_size_preds = torch.cat(all_size_preds).numpy()
    all_type_labels = torch.cat(all_type_labels).numpy()
    all_size_labels = torch.cat(all_size_labels).numpy()
    return (total_loss / n, type_acc.compute().item(), size_acc.compute().item(),
            all_type_preds, all_size_preds, all_type_labels, all_size_labels)


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print('=' * 60)
    print('CWRU CNN-2D Retrain with Best Params')
    print('=' * 60)
    print(f'  n_fft: {N_FFT}, hop_length: {HOP_LENGTH}, freq_bins: {FREQ_BINS}')
    print(f'  频率分辨率: {48000/N_FFT:.1f} Hz')
    print(f'  时间分辨率: {HOP_LENGTH/48000*1000:.2f} ms')
    print()

    if not os.path.exists(BEST_PARAMS_PATH):
        print('Error: best_params_2d.json not found. Run search first.')
        return

    with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
        best_data = json.load(f)
    params = best_data['params']
    chs = []
    for i in range(params['num_conv_layers']):
        ch = min(int(params['base_channels'] * (params['channel_growth'] ** i)), 512)
        chs.append(ch)
    params['channels'] = chs
    batch_size = params['batch_size']
    print('Best trial: #' + str(best_data['trial_number']))
    print('Val type_acc: ' + str(best_data.get('val_type_acc', 'N/A')))
    print('Val size_acc: ' + str(best_data.get('val_size_acc', 'N/A')))
    print('Best params: ' + str(params))
    print()

    set_seed(RANDOM_SEED)
    model = build_model(params).to(DEVICE)
    n_params = model.count_params()
    print('Model params: ' + str(n_params))

    criterion = nn.CrossEntropyLoss()
    train_ds = CWRUDataset('train', add_noise_snr=ADD_NOISE_SNR, normalize=NORMALIZE)
    val_ds = CWRUDataset('val', normalize=NORMALIZE)
    test_ds = CWRUDataset('test', normalize=NORMALIZE)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    print('Train/Val/Test samples: ' + str(len(train_ds)) + ' / ' + str(len(val_ds)) + ' / ' + str(len(test_ds)))

    ckpt = os.path.exists('code4/outputs/best_model_2d.pth')
    if ckpt:
        best_state = torch.load('code4/outputs/best_model_2d.pth', map_location=DEVICE)
        print("加载已经存在的权重！")
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=RETRAIN_LR_FACTOR, patience=RETRAIN_LR_PATIENCE)

        best_val_joint = -1.0
        best_state = None
        patience_counter = 0
        print('\nTraining...')
        for epoch in range(RETRAIN_EPOCHS):
            train_loss, train_type, train_size = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_type, val_size, _, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            val_joint = (val_type + val_size) / 2.0
            scheduler.step(val_loss)
            if val_joint > best_val_joint:
                best_val_joint = val_joint
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print('Epoch %d/%d - Train loss: %.4f, Val loss: %.4f, Val joint: %.4f' % (epoch + 1, RETRAIN_EPOCHS, train_loss, val_loss, val_joint))
            if patience_counter >= RETRAIN_PATIENCE:
                print('Early stopping at epoch ' + str(epoch + 1))
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUTPUTS_DIR, 'best_model_2d.pth'))
    print('Model saved to best_model_2d.pth')

    # Test evaluation
    print('\nEvaluating on test set...')
    test_loss, test_type_acc, test_size_acc, type_preds, size_preds, type_labels, size_labels = evaluate(model, test_loader, criterion, DEVICE)
    test_joint = (test_type_acc + test_size_acc) / 2.0

    print('\n' + '============================================================')
    print('Test Results (CNN-2D)')
    print('============================================================')
    print('Test Type Acc: %.4f' % test_type_acc)
    print('Test Size Acc: %.4f' % test_size_acc)
    print('Test Joint Acc: %.4f' % test_joint)
    print()

    # Classification reports
    print('Type Classification Report:')
    print(classification_report(type_labels, type_preds, target_names=TYPE_NAMES, digits=4))
    print('Size Classification Report:')
    print(classification_report(size_labels, size_preds, target_names=SIZE_NAMES, digits=4))

    # Confusion matrices
    cm_type = confusion_matrix(type_labels, type_preds)
    cm_size = confusion_matrix(size_labels, size_preds)
    print('Type Confusion Matrix:')
    print(cm_type)
    print('Size Confusion Matrix:')
    print(cm_size)

    plot_confusion_matrix(cm_type, TYPE_NAMES, 'CWRU CNN-2D Test - Fault Type Confusion Matrix', os.path.join(OUTPUTS_DIR, 'cm_type_2d.png'))
    plot_confusion_matrix(cm_size, SIZE_NAMES, 'CWRU CNN-2D Test - Fault Size Confusion Matrix', os.path.join(OUTPUTS_DIR, 'cm_size_2d.png'))

    print('\nResults saved to outputs/ directory.')


if __name__ == '__main__':
    main()
