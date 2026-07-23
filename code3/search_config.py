import sys
import os

# 将 code2 加入路径，复用其 config 和 dataset
CODE2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code2')
sys.path.insert(0, CODE2_DIR)

from config import (
    ROOT_DIR, RAW_DATA_DIR, MAPPING_FILE, OUTPUT_DIR,
    WINDOW_LENGTH, OVERLAP_RATIO, RANDOM_SEED,
    TYPE_MAP, SIZE_MAP, FS,
    TRAIN_FILES, VAL_FILES, TEST_FILES,
    ADD_NOISE_SNR, NORMALIZE,
)

# ==================== 搜索控制参数 ====================
N_TRIALS = 2               # Optuna 搜索总试验次数
N_EPOCHS_PER_TRIAL = 120     # 每个 trial 最大训练 epoch 数
EARLY_STOP_PATIENCE = 15    # 每个 trial 内的早停 patience
REDUCE_LR_PATIENCE = 8      # 学习率衰减 patience
REDUCE_LR_FACTOR = 0.5      # 学习率衰减因子

# 多目标优化方向
OBJECTIVE_DIRECTIONS = ['maximize', 'maximize']  # [type_acc, size_acc]

# 固定参数（不参与搜索）
INPUT_LENGTH = WINDOW_LENGTH
NUM_TYPE_CLASSES = len(TYPE_MAP)
NUM_SIZE_CLASSES = len(SIZE_MAP)

# ==================== Optuna 搜索空间定义 ====================

def suggest_params(trial):
    params = {}
    # 卷积层数：3~6 层
    params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 3, 6)
    # 第一层通道基数
    base_channels = trial.suggest_categorical('base_channels', [16, 32, 48, 64])
    # 通道增长因子（>=1.0，保证通道数不减少）
    params['channel_growth'] = trial.suggest_float('channel_growth', 1.0, 2.5, step=0.25)
    # 生成各层通道数列表
    channels = []
    for i in range(params['num_conv_layers']):
        ch = int(base_channels * (params['channel_growth'] ** i))
        ch = min(ch, 512)  # 限制最大通道数
        channels.append(ch)
    params['channels'] = channels
    # 第一层宽核参数
    params['first_kernel'] = trial.suggest_categorical('first_kernel', [16, 32, 64, 128])
    params['first_stride'] = trial.suggest_categorical('first_stride', [4, 8, 16])
    # 后续层卷积核大小
    params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
    # 池化
    params['use_pool'] = trial.suggest_categorical('use_pool', [True, False])
    if params['use_pool']:
        params['pool_size'] = trial.suggest_categorical('pool_size', [2, 4])
        params['pool_stride'] = trial.suggest_categorical('pool_stride', [2, 4])
    # 分类头
    params['type_hidden'] = trial.suggest_categorical('type_hidden', [32, 64, 96, 128])
    params['size_hidden'] = trial.suggest_categorical('size_hidden', [32, 64, 96, 128])
    params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    # 优化器
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
    return params


def compute_encoder_output_dim(params, input_length=INPUT_LENGTH):
    L = input_length
    channels = params['channels']
    for i in range(params['num_conv_layers']):
        if i == 0:
            kernel = params['first_kernel']
            stride = params['first_stride']
            L = (L - kernel) // stride + 1
        if params.get('use_pool', False):
            pool_size = params.get('pool_size', 2)
            pool_stride = params.get('pool_stride', 2)
            L = (L - pool_size) // pool_stride + 1
        if L < 1:
            L = 1
    return channels[-1]


if __name__ == '__main__':
    import optuna
    def objective(trial):
        params = suggest_params(trial)
        feat_dim = compute_encoder_output_dim(params)
        print('Trial sample: layers=' + str(params['num_conv_layers']) + ' channels=' + str(params['channels']) + ' feat_dim=' + str(feat_dim))
        return 0.0
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)
    print('Sampling test passed.')
