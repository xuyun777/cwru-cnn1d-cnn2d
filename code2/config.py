# ============================================================
# config.py -- CWRU 轴承故障诊断 参数集中管理（code2版）
#
# 核心改进：按负载（HP）划分数据集，消除滑窗数据泄漏。
#   - 训练集：0HP + 1HP 所有样本
#   - 验证集：2HP 所有样本
#   - 测试集：3HP 所有样本
# 每个 .mat 文件的所有窗口都在同一个划分中。
# ============================================================

import os
import json

# ==================== 路径配置 ====================
# 项目根目录（code2/ 的上级）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始 .mat 文件存放目录
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')

# 文件与标签对应关系 JSON
MAPPING_FILE = os.path.join(RAW_DATA_DIR, 'index_table.json')

# 预处理输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'output')

# ==================== 滑窗切片参数 ====================
WINDOW_LENGTH = 8192        # 每个样本的点数
OVERLAP_RATIO = 0.5         # 相邻窗口重叠比例
# 实际步长 = WINDOW_LENGTH * (1 - OVERLAP_RATIO) = 1024

# ==================== 数据集划分参数 ====================
RANDOM_SEED = 42

# ==================== 标签映射 ====================
# 故障类型标签：0=正常, 1=内圈(IR), 2=外圈(OR), 3=滚动体(ball)
TYPE_MAP = {
    'normal': 0,
    'IR': 1,
    'OR': 2,
    'ball': 3,
}

# 故障尺寸标签：0=无故障, 1=0.007inch, 2=0.014inch, 3=0.021inch
SIZE_MAP = {
    'none': 0,
    '007': 1,
    '014': 2,
    '021': 3,
}

# ==================== 信号采样率 ====================
FS = 48000  # 48 kHz

# ==================== 输出文件名 ====================
TRAIN_X_FILE = 'train_X.npy'
TRAIN_TYPE_FILE = 'train_type.npy'
TRAIN_SIZE_FILE = 'train_size.npy'
VAL_X_FILE = 'val_X.npy'
VAL_TYPE_FILE = 'val_type.npy'
VAL_SIZE_FILE = 'val_size.npy'
TEST_X_FILE = 'test_X.npy'
TEST_TYPE_FILE = 'test_type.npy'
TEST_SIZE_FILE = 'test_size.npy'
STATS_FILE = 'preprocess_stats.txt'

# ==================== 训练超参数 ====================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 120
PATIENCE = 20
LR_PATIENCE = 10
LR_FACTOR = 0.5
DROPOUT_RATE = 0.3
MODEL_SAVE_NAME = 'best_model.pth'


def _parse_key(key):
    parts = key.split('_')
    if parts[0] == 'normal':
        fault_type = 'normal'
        size = 'none'
        load = int(parts[1])
    else:
        size = parts[0]
        fault_type = parts[1]
        load = int(parts[2]) - 1
    return fault_type, size, load


def build_file_lists():
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(
            'Mapping file not found: ' + MAPPING_FILE
        )
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    train_files = []
    val_files = []
    test_files = []

    for key, filename in mapping.items():
        fault_type_str, size_str, load = _parse_key(key)
        type_label = TYPE_MAP[fault_type_str]
        size_label = SIZE_MAP[size_str]
        entry = (filename, type_label, size_label)
        if load in (0, 1):
            train_files.append(entry)
        elif load == 2:
            val_files.append(entry)
        elif load == 3:
            test_files.append(entry)
        else:
            raise ValueError('Unknown load: ' + str(load) + ' key=' + key)

    return train_files, val_files, test_files


TRAIN_FILES, VAL_FILES, TEST_FILES = build_file_lists()

if __name__ == '__main__':
    print(f'Train files: {len(TRAIN_FILES)}')
    print(f'Val files: {len(VAL_FILES)}')
    print(f'Test files: {len(TEST_FILES)}')
    for fname, t, s in TRAIN_FILES[:3]:
        print(f'  train: {fname} type={t} size={s}')
    for fname, t, s in VAL_FILES[:3]:
        print(f'  val:   {fname} type={t} size={s}')
    for fname, t, s in TEST_FILES[:3]:
        print(f'  test:  {fname} type={t} size={s}')
# ==================== 数据增强与归一化 ====================
# 训练时注入多级高斯白噪声（仅在训练集生效）
# 单值 float：固定 SNR；列表：每个样本随机选取一个 SNR
ADD_NOISE_SNR = [20, 30, 40]  # SNR 列表（dB）
# 逐样本归一化方式：standard / minmax / None
NORMALIZE = 'standard'
