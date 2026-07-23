# ============================================================
# preprocess.py -- CWRU 轴承数据预处理（code2版）
#
# 核心改进：按负载（HP）划分数据集，而非随机划分。
#   训练集 = 0HP + 1HP 所有文件
#   验证集 = 2HP 所有文件
#   测试集 = 3HP 所有文件
# 每个 .mat 文件的所有滑窗完整保留在同一个划分中，
# 避免相邻窗口跨越训练/验证集造成数据泄漏。
#
# 运行方式：python code2/preprocess.py
# ============================================================

import numpy as np
import scipy.io as sio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RAW_DATA_DIR, OUTPUT_DIR,
    WINDOW_LENGTH, OVERLAP_RATIO,
    TRAIN_FILES, VAL_FILES, TEST_FILES,
    TRAIN_X_FILE, TRAIN_TYPE_FILE, TRAIN_SIZE_FILE,
    VAL_X_FILE, VAL_TYPE_FILE, VAL_SIZE_FILE,
    TEST_X_FILE, TEST_TYPE_FILE, TEST_SIZE_FILE,
    STATS_FILE,
)


def sliding_window(signal, window_length, overlap_ratio):
    step = int(window_length * (1 - overlap_ratio))
    if step <= 0:
        raise ValueError(
            'Overlap ratio ' + str(overlap_ratio) + ' too large, step <= 0'
        )
    n_samples = (len(signal) - window_length) // step + 1
    segments = np.zeros((n_samples, window_length), dtype=np.float32)
    for i in range(n_samples):
        start = i * step
        segments[i] = signal[start:start + window_length]
    return segments


def load_and_slice(mat_filename):
    filepath = os.path.join(RAW_DATA_DIR, mat_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError('File not found: ' + filepath)
    mat = sio.loadmat(filepath)
    file_id = mat_filename.replace('.mat', '').zfill(3)
    de_key = 'X' + file_id + '_DE_time'
    if de_key not in mat:
        available_keys = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(
            'Key ' + de_key + ' not found in ' + mat_filename
            + '. Available keys: ' + str(available_keys)
        )
    signal = mat[de_key].flatten().astype(np.float32)
    segments = sliding_window(signal, WINDOW_LENGTH, OVERLAP_RATIO)
    return segments


def process_split(file_list, split_name):
    all_samples = []
    all_type_labels = []
    all_size_labels = []

    for fname, type_label, size_label in file_list:
        try:
            segments = load_and_slice(fname)
        except (FileNotFoundError, KeyError) as e:
            print('  [WARNING] ' + str(e))
            continue

        n = len(segments)
        all_samples.append(segments)
        all_type_labels.append(np.full(n, type_label, dtype=np.int64))
        all_size_labels.append(np.full(n, size_label, dtype=np.int64))
        print('  ' + fname + ': ' + str(n) + ' samples (type=' + str(type_label) + ', size=' + str(size_label) + ')')

    if not all_samples:
        print('  [ERROR] No data for ' + split_name)
        return None, None, None

    X = np.concatenate(all_samples, axis=0)
    y_type = np.concatenate(all_type_labels, axis=0)
    y_size = np.concatenate(all_size_labels, axis=0)
    return X, y_type, y_size


def main():
    print('=' * 60)
    print('CWRU Bearing Fault Diagnosis -- Data Preprocessing (code2)')
    print('=' * 60)

    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print('Created data directory: ' + RAW_DATA_DIR)
        print('Please put .mat files in data/raw/ and re-run.')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print('[1/3] Processing training set (0HP + 1HP)...')
    X_train, y_type_train, y_size_train = process_split(TRAIN_FILES, 'train')
    if X_train is not None:
        print('  Total train: ' + str(len(X_train)) + ' samples, shape=' + str(X_train.shape))

    print()
    print('[2/3] Processing validation set (2HP)...')
    X_val, y_type_val, y_size_val = process_split(VAL_FILES, 'val')
    if X_val is not None:
        print('  Total val: ' + str(len(X_val)) + ' samples, shape=' + str(X_val.shape))

    print()
    print('[3/3] Processing test set (3HP)...')
    X_test, y_type_test, y_size_test = process_split(TEST_FILES, 'test')
    if X_test is not None:
        print('  Total test: ' + str(len(X_test)) + ' samples, shape=' + str(X_test.shape))

    print()
    print('Saving to ' + OUTPUT_DIR + ' ...')

    np.save(os.path.join(OUTPUT_DIR, TRAIN_X_FILE), X_train)
    np.save(os.path.join(OUTPUT_DIR, TRAIN_TYPE_FILE), y_type_train)
    np.save(os.path.join(OUTPUT_DIR, TRAIN_SIZE_FILE), y_size_train)
    np.save(os.path.join(OUTPUT_DIR, VAL_X_FILE), X_val)
    np.save(os.path.join(OUTPUT_DIR, VAL_TYPE_FILE), y_type_val)
    np.save(os.path.join(OUTPUT_DIR, VAL_SIZE_FILE), y_size_val)
    np.save(os.path.join(OUTPUT_DIR, TEST_X_FILE), X_test)
    np.save(os.path.join(OUTPUT_DIR, TEST_TYPE_FILE), y_type_test)
    np.save(os.path.join(OUTPUT_DIR, TEST_SIZE_FILE), y_size_test)

    stats_path = os.path.join(OUTPUT_DIR, STATS_FILE)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('CWRU Preprocessing Stats (code2) -- Per-Load Split\n')
        f.write('=' * 50 + '\n\n')
        f.write('Window length: ' + str(WINDOW_LENGTH) + '\n')
        f.write('Overlap ratio: ' + str(OVERLAP_RATIO) + '\n')
        f.write('Train (0HP+1HP): ' + str(len(X_train)) + ' samples\n')
        f.write('Val (2HP): ' + str(len(X_val)) + ' samples\n')
        f.write('Test (3HP): ' + str(len(X_test)) + ' samples\n')
        type_names = {0: 'Normal', 1: 'Inner', 2: 'Outer', 3: 'Ball'}
        for name, y in [('Train', y_type_train), ('Val', y_type_val), ('Test', y_type_test)]:
            f.write('\n' + name + ' type distribution:\n')
            for label, count in zip(*np.unique(y, return_counts=True)):
                f.write('  ' + type_names.get(label, str(label)) + ': ' + str(count) + '\n')

    print('Stats saved to: ' + stats_path)
    print()
    print('=' * 60)
    print('Preprocessing complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
