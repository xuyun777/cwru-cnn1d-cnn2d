import sys, os
CODE2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code2')
sys.path.insert(0, CODE2_DIR)
from config import (ROOT_DIR, RAW_DATA_DIR, MAPPING_FILE, OUTPUT_DIR, WINDOW_LENGTH, OVERLAP_RATIO, TYPE_MAP, SIZE_MAP, FS, TRAIN_FILES, VAL_FILES, TEST_FILES)

N_TRIALS = 4
N_EPOCHS_PER_TRIAL = 120
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5
OBJECTIVE_DIRECTIONS = ['maximize', 'maximize']
INPUT_LENGTH = WINDOW_LENGTH          # 与 config.py WINDOW_LENGTH 一致
NUM_TYPE_CLASSES = len(TYPE_MAP)
NUM_SIZE_CLASSES = len(SIZE_MAP)
N_FFT = 1024                 # 频率分辨率 df = 48000/1024 = 46.9 Hz
HOP_LENGTH = 256             # 时间分辨率 dt = 256/48000 = 5.3 ms
SPEC_SIZE = 200              # 频率bin数（FREQ_BINS），覆盖 0-9.4 kHz
FREQ_BINS = 200
# 逐样本归一化方式：standard / minmax / None
NORMALIZE = 'standard'
RANDOM_SEED = 42
ADD_NOISE_SNR = [20, 30, 40]  # SNR 列表（dB）
FS = FS


def suggest_params(trial):
    """从 Optuna trial 采样 CNN-2D 超参数"""
    params = {}
    params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 3, 6)
    params['base_channels'] = trial.suggest_categorical('base_channels', [16, 32, 48, 64])
    params['channel_growth'] = trial.suggest_float('channel_growth', 1.0, 2.5, step=0.25)
    chs = []
    for i in range(params['num_conv_layers']):
        ch = min(int(params['base_channels'] * (params['channel_growth'] ** i)), 512)
        chs.append(ch)
    params['channels'] = chs
    params['first_kernel'] = trial.suggest_categorical('first_kernel', [3, 5, 7])
    params['first_stride'] = trial.suggest_categorical('first_stride', [2, 4])
    params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5])
    params['use_pool'] = trial.suggest_categorical('use_pool', [True, False])
    if params['use_pool']:
        params['pool_size'] = trial.suggest_categorical('pool_size', [2, 4])
    params['type_hidden'] = trial.suggest_categorical('type_hidden', [32, 64, 96, 128])
    params['size_hidden'] = trial.suggest_categorical('size_hidden', [32, 64, 96, 128])
    params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
    return params
