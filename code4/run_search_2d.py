import optuna
import sys, os, json, csv
from datetime import datetime
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from search_config_2d import N_TRIALS, OBJECTIVE_DIRECTIONS, RANDOM_SEED
from search_train_2d import objective
OUTPUTS_DIR = os.path.join(CODE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def select_best_from_pareto(study):
    pareto_trials = [t for t in study.best_trials]
    best_trial, best_joint = None, -1.0
    for t in pareto_trials:
        joint = t.user_attrs.get('best_val_joint', 0.0)
        if joint > best_joint:
            best_joint, best_trial = joint, t
    return best_trial

def save_pareto_front(study, filepath):
    pareto_trials = [t for t in study.best_trials]
    all_keys = set()
    for t in pareto_trials:
        all_keys.update(t.params.keys())
    param_keys = sorted(all_keys)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        base = 'trial_number,val_type_acc,val_size_acc,val_joint_acc,n_params'.split(',')
        writer.writerow(base + param_keys)
        for t in sorted(pareto_trials, key=lambda x: x.user_attrs.get('best_val_joint', 0), reverse=True):
            row = [t.number, round(t.values[0], 6), round(t.values[1], 6), round(t.user_attrs.get('best_val_joint', 0), 6), t.user_attrs.get('n_params', 0)]
            for k in param_keys:
                row.append(t.params.get(k, ''))
            writer.writerow(row)
    print('Pareto front saved: ' + filepath)

def save_all_trials(study, filepath):
    all_keys = set()
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            all_keys.update(t.params.keys())
    param_keys = sorted(all_keys)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        base = 'trial_number,state,val_type_acc,val_size_acc,val_joint_acc,n_params'.split(',')
        writer.writerow(base + param_keys)
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                row = [t.number, 'COMPLETE', round(t.values[0], 6), round(t.values[1], 6), round(t.user_attrs.get('best_val_joint', 0), 6), t.user_attrs.get('n_params', 0)]
                for k in param_keys:
                    row.append(t.params.get(k, ''))
                writer.writerow(row)
    print('All trials saved: ' + filepath)

def main():
    print('============================================================')
    print('CWRU CNN-2D Multi-Objective Hyperparameter Search')
    print('============================================================')
    print(datetime.now().strftime('Start: %Y-%m-%d %H:%M:%S'))
    print(f'N_TRIALS: {N_TRIALS}')
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True)
    storage = f"sqlite:///{OUTPUTS_DIR}/study_SQLite_2d.db"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(directions=OBJECTIVE_DIRECTIONS, sampler=sampler, pruner=pruner,
                                storage=storage,study_name='cwru_cnn2d',load_if_exists=True)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print('\nSearch completed!')
    best_trial = select_best_from_pareto(study)
    if best_trial is not None:
        print(f'Best Trial: type_acc={best_trial.values[0]:.6f} size_acc={best_trial.values[1]:.6f}')
        best_save = {'trial_number': best_trial.number, 'val_type_acc': best_trial.values[0], 'val_size_acc': best_trial.values[1], 'val_joint_acc': best_trial.user_attrs.get('best_val_joint', 0), 'n_params': best_trial.user_attrs.get('n_params', 0), 'params': best_trial.params}
        with open(os.path.join(OUTPUTS_DIR, 'best_params_2d.json'), 'w', encoding='utf-8') as f:
            json.dump(best_save, f, indent=2, ensure_ascii=False)
    save_pareto_front(study, os.path.join(OUTPUTS_DIR, 'pareto_front_2d.csv'))
    save_all_trials(study, os.path.join(OUTPUTS_DIR, 'all_trials_2d.csv'))
    print('Done!')

if __name__ == '__main__':
    main()
