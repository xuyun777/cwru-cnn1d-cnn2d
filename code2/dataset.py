import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    OUTPUT_DIR,
    TRAIN_X_FILE, TRAIN_TYPE_FILE, TRAIN_SIZE_FILE,
    VAL_X_FILE, VAL_TYPE_FILE, VAL_SIZE_FILE,
    TEST_X_FILE, TEST_TYPE_FILE, TEST_SIZE_FILE,
)


class CWRUDataset(Dataset):

    def __init__(self, split='train', add_noise_snr=None, normalize='standard'):
        if split not in ['train', 'val', 'test']:
            raise ValueError('split must be train/val/test, got: ' + split)
        self.split = split
        self.normalize = normalize

        if split == 'train':
            x_file, type_file, size_file = TRAIN_X_FILE, TRAIN_TYPE_FILE, TRAIN_SIZE_FILE
        elif split == 'val':
            x_file, type_file, size_file = VAL_X_FILE, VAL_TYPE_FILE, VAL_SIZE_FILE
        else:
            x_file, type_file, size_file = TEST_X_FILE, TEST_TYPE_FILE, TEST_SIZE_FILE

        self.X_raw = np.load(os.path.join(OUTPUT_DIR, x_file))
        self.y_type = np.load(os.path.join(OUTPUT_DIR, type_file))
        self.y_size = np.load(os.path.join(OUTPUT_DIR, size_file))

        if split == 'train' and add_noise_snr is not None:
            self.add_noise_snr = add_noise_snr
            self._global_power = np.mean(self.X_raw ** 2)
        else:
            self.add_noise_snr = None
            self._global_power = None

    def _add_noise(self, x):
        if isinstance(self.add_noise_snr, (list, tuple)):
            snr = np.random.choice(self.add_noise_snr)
        else:
            snr = self.add_noise_snr

        p_signal = np.mean(x ** 2)
        if p_signal <= 0:
            return x
        p_noise = p_signal / (10 ** (snr / 10))
        noise = np.random.randn(*x.shape).astype(np.float32) * np.sqrt(p_noise)
        return x + noise

    def _normalize_sample(self, x):
        if self.normalize == 'standard':
            mu = np.mean(x)
            sigma = np.std(x)
            if sigma < 1e-8:
                return x - mu
            return (x - mu) / sigma
        elif self.normalize == 'minmax':
            x_min = np.min(x)
            x_max = np.max(x)
            denom = x_max - x_min
            if denom < 1e-8:
                return x - x_min
            return (x - x_min) / denom
        else:
            return x

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        x = self.X_raw[idx].copy()

        if self.add_noise_snr is not None:
            x = self._add_noise(x)

        if self.normalize is not None:
            x = self._normalize_sample(x)

        x = x[np.newaxis, :]

        waveform = torch.from_numpy(x).float()
        type_label = torch.tensor(self.y_type[idx], dtype=torch.long)
        size_label = torch.tensor(self.y_size[idx], dtype=torch.long)
        return waveform, type_label, size_label

    def get_class_counts(self, label_type='type'):
        y = self.y_type if label_type == 'type' else self.y_size
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))


def create_dataloaders(batch_size=64, num_workers=0, add_noise_snr=None, normalize='standard'):
    train_ds = CWRUDataset('train', add_noise_snr=add_noise_snr, normalize=normalize)
    val_ds = CWRUDataset('val', normalize=normalize)
    test_ds = CWRUDataset('test', normalize=normalize)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        snr = 30 if split == 'train' else None
        ds = CWRUDataset(split, add_noise_snr=snr, normalize='standard')
        print(split + ': ' + str(len(ds)) + ' samples')
        print('  type dist: ' + str(ds.get_class_counts('type')))
        x, t, s = ds[0]
        print('  shape: ' + str(x.shape) + ', range: [' + str(round(x.min().item(), 4)) + ', ' + str(round(x.max().item(), 4)) + '], type=' + str(t.item()) + ', size=' + str(s.item()))
        print()
