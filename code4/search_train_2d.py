import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import sys, os, time, random
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.join(os.path.dirname(CODE_DIR), 'code2')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CODE2_DIR)

from search_config_2d import (suggest_params, INPUT_LENGTH, NUM_TYPE_CLASSES,
                               NUM_SIZE_CLASSES, N_EPOCHS_PER_TRIAL,
                               EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
                               REDUCE_LR_FACTOR, ADD_NOISE_SNR, NORMALIZE,
                               RANDOM_SEED)
from search_model_2d import build_model 
from dataset import CWRUDataset
import stft_transform_2d as stft_transform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    return {'loss': total_loss / n, 'type_acc': type_acc.compute().item(), 'size_acc': size_acc.compute().item()}


@torch.no_grad()
def validate(model, loader, criterion, type_acc, size_acc, device):
    model.eval()
    total_loss = 0.0
    for waveforms, type_labels, size_labels in loader:
        waveforms = waveforms.to(device)
        type_labels = type_labels.to(device)
        size_labels = size_labels.to(device)
        specs = stft_transform.batch_to_spectrograms(waveforms).to(device)
        type_logits, size_logits = model(specs)
        loss = criterion(type_logits, type_labels) + criterion(size_logits, size_labels)
        total_loss += loss.item()
        type_acc.update(type_logits, type_labels)
        size_acc.update(size_logits, size_labels)
    n = len(loader)
    return {'loss': total_loss / n, 'type_acc': type_acc.compute().item(), 'size_acc': size_acc.compute().item()}


def objective(trial):

    mins = 0
    if mins:
        print(f"暂停{mins}分钟")
        time.sleep(mins*60)

    # Optuna multi-objective: maximize both val_type_acc and val_size_acc
    set_seed(RANDOM_SEED)
    params = suggest_params(trial)
    batch_size = params['batch_size']

    try:
        model = build_model(params).to(DEVICE)
    except Exception as e:
        return 0, 0
    n_params = model.count_params()

    train_ds = CWRUDataset('train', add_noise_snr=ADD_NOISE_SNR, normalize=NORMALIZE)
    val_ds = CWRUDataset('val', normalize=NORMALIZE)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE)

    type_acc_metric = Accuracy(task='multiclass', num_classes=NUM_TYPE_CLASSES).to(DEVICE)
    size_acc_metric = Accuracy(task='multiclass', num_classes=NUM_SIZE_CLASSES).to(DEVICE)

    best_val_joint = -1.0
    best_vals = (0.0, 0.0)
    patience_counter = 0

    for epoch in range(N_EPOCHS_PER_TRIAL):
        type_acc_metric.reset()
        size_acc_metric.reset()
        train_one_epoch(model, train_loader, criterion, optimizer, type_acc_metric, size_acc_metric, DEVICE)

        type_acc_metric.reset()
        size_acc_metric.reset()
        val_result = validate(model, val_loader, criterion, type_acc_metric, size_acc_metric, DEVICE)

        val_joint = 0.4*val_result['type_acc'] + 0.6*val_result['size_acc']
        scheduler.step(val_result['loss'])

        if val_joint > best_val_joint:
            best_val_joint = val_joint
            best_vals = (val_result['type_acc'], val_result['size_acc'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break
    trial.set_user_attr('best_vals', best_vals)
    trial.set_user_attr('best_val_joint', best_val_joint)
    trial.set_user_attr('n_params', n_params)
    return best_vals[0], best_vals[1]
