
import optuna
import sys
import os
import json
import csv
from datetime import datetime

# 将 code3 加入路径
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from search_config import (
    N_TRIALS, OBJECTIVE_DIRECTIONS, RANDOM_SEED,
    N_EPOCHS_PER_TRIAL, EARLY_STOP_PATIENCE,
)
from search_train import objective

# 输出目录
OUTPUTS_DIR = os.path.join(CODE_DIR, 'outputs')


def select_best_from_pareto(study):
    pareto_trials = [t for t in study.best_trials]
    best_trial = None
    best_joint = -1.0

    for t in pareto_trials:
        joint = t.user_attrs.get('best_val_joint', 0.0)
        if joint > best_joint:
            best_joint = joint
            best_trial = t

    return best_trial


def save_pareto_front(study, filepath):
    pareto_trials = [t for t in study.best_trials]

    # 收集所有超参数名称
    all_param_keys = set()
    for t in pareto_trials:
        all_param_keys.update(t.params.keys())
    param_keys = sorted(all_param_keys)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['trial_number', 'val_type_acc', 'val_size_acc', 'val_joint_acc', 'n_params'] + param_keys
        writer.writerow(header)

        for t in sorted(pareto_trials, key=lambda x: x.user_attrs.get('best_val_joint', 0), reverse=True):
            row = [
                t.number,
                round(t.values[0], 6),
                round(t.values[1], 6),
                round(t.user_attrs.get('best_val_joint', 0), 6),
                t.user_attrs.get('n_params', 0),
            ]
            for k in param_keys:
                row.append(t.params.get(k, ''))
            writer.writerow(row)

    print('Pareto front saved: ' + filepath)
    print('  Number of Pareto-optimal solutions: ' + str(len(pareto_trials)))


def save_all_trials(study, filepath):
    """保存所有 trial 记录到 CSV 文件。"""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    all_param_keys = set()
    for t in completed:
        all_param_keys.update(t.params.keys())
    param_keys = sorted(all_param_keys)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['trial_number', 'state', 'val_type_acc', 'val_size_acc', 'val_joint_acc', 'n_params', 'duration_sec'] + param_keys
        writer.writerow(header)

        for t in completed:
            duration = (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else 0
            row = [
                t.number,
                str(t.state),
                round(t.values[0], 6) if t.values else '',
                round(t.values[1], 6) if t.values else '',
                round(t.user_attrs.get('best_val_joint', 0), 6),
                t.user_attrs.get('n_params', 0),
                round(duration, 1),
            ]
            for k in param_keys:
                row.append(t.params.get(k, ''))
            writer.writerow(row)

    print('All trials saved: ' + filepath)
    print('  Completed trials: ' + str(len(completed)))


def save_best_params(best_trial, filepath):
    """保存最佳超参数到 JSON 文件。"""
    result = {
        'trial_number': best_trial.number,
        'val_type_acc': round(best_trial.values[0], 6),
        'val_size_acc': round(best_trial.values[1], 6),
        'val_joint_acc': round(best_trial.user_attrs.get('best_val_joint', 0), 6),
        'n_params': best_trial.user_attrs.get('n_params', 0),
        'params': best_trial.params,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print('Best params saved: ' + filepath)


def print_summary(study, best_trial):
    """打印搜索摘要。"""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    pareto_count = len(study.best_trials)

    print()
    print('=' * 70)
    print('  OPTUNA SEARCH SUMMARY')
    print('=' * 70)
    print('  Start time:    ' + study.trials[0].datetime_start.strftime('%Y-%m-%d %H:%M:%S') if study.trials else 'N/A')
    print('  Total trials:  ' + str(len(study.trials)))
    print('  Completed:     ' + str(len(completed)))
    print('  Pruned:        ' + str(len(pruned)))
    print('  Pareto front:  ' + str(pareto_count) + ' solutions')
    print()
    print('  Best (by joint_acc on Pareto front):')
    print('    Trial #' + str(best_trial.number))
    print('    val_type_acc:  ' + str(round(best_trial.values[0], 4)))
    print('    val_size_acc:  ' + str(round(best_trial.values[1], 4)))
    print('    val_joint_acc: ' + str(round(best_trial.user_attrs.get('best_val_joint', 0), 4)))
    print('    n_params:      ' + str(best_trial.user_attrs.get('n_params', 0)))
    print()
    print('  Best hyperparameters:')
    for k, v in sorted(best_trial.params.items()):
        print('    ' + k + ' = ' + str(v))
    print('=' * 70)


def main():
    print('=' * 70)
    print('CWRU Bearing Fault Diagnosis -- Optuna Multi-Objective Search (code3)')
    print('Start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Directions: ' + str(OBJECTIVE_DIRECTIONS))
    print('Max trials: ' + str(N_TRIALS))
    print('Max epochs/trial: ' + str(N_EPOCHS_PER_TRIAL))
    print('Early stop patience: ' + str(EARLY_STOP_PATIENCE))
    print('=' * 70)

    # 创建输出目录
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ---- 创建多目标 study ----
    # 使用 TPE Sampler（与直升机项目一致），Median Pruner 用于中间剪枝
    storage = f"sqlite:///{OUTPUTS_DIR}/study_SQLite.db"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True)
    study = optuna.create_study(
        directions=OBJECTIVE_DIRECTIONS,
        sampler=sampler,
        study_name='cwru_cnn1d_multiob',
        storage=storage,
        load_if_exists=True
    )

    # ---- 运行搜索 ----
    print()
    print('Starting search...')
    print('-' * 70)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ---- 选最优解 ----
    best_trial = select_best_from_pareto(study)

    if best_trial is None:
        print('ERROR: No valid trials found. All trials may have failed.')
        return

    # ---- 保存结果 ----
    print()
    print('Saving results...')

    save_best_params(best_trial, os.path.join(OUTPUTS_DIR, 'best_params.json'))
    save_pareto_front(study, os.path.join(OUTPUTS_DIR, 'pareto_front.csv'))
    save_all_trials(study, os.path.join(OUTPUTS_DIR, 'all_trials.csv'))

    # ---- 打印摘要 ----
    print_summary(study, best_trial)

    print()
    print('Search complete! End: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
