import os
import sys
import itertools
import logging
import joblib
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .set_seeds import set_seeds
from .augment_train import augment_train
from .train_single_model import train_single_model
from .fit_calibrators import fit_calibrators
from .apply_calibrators import apply_calibrators
from .threshold_tuning import threshold_tuning
from .compute_metrics_and_save import compute_metrics_and_save
from .plot_ensemble_figures import plot_ensemble_figures

# Get the absolute path of the directory above the current working directory
module_path = os.path.abspath(os.path.join('..'))
# Insert the path into sys.path at index 0 to ensure it is checked first
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from cls.Preprocessor import Preprocessor
from cls.EnsemblePredictor import EnsemblePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ensemble_pipeline(X_train, Y_train, X_test, Y_test, cfg):
    # Extract classes directly from the config
    class_list = cfg["default_classes"]
    
    set_seeds(cfg['global_seed'])
    root = "./" 
    ensemble_root = os.path.join(root, cfg['ensemble_sub_dir'])
    dir_train_report = os.path.join(ensemble_root, "train_cv_report")
    dir_test_report = os.path.join(ensemble_root, "test_report")

    os.makedirs(ensemble_root, exist_ok=True)
    os.makedirs(dir_train_report, exist_ok=True)
    os.makedirs(dir_test_report, exist_ok=True)

    ensemble_artifacts = {'models': [], 'preps': [], 'calibs': [], 'thresh': [], 'pos_weights': []}
    oof_preds = np.full((X_train.shape[0], len(class_list)), -1.0)

    param_grid = list(itertools.product(
        cfg['lr_options'], cfg['weight_decay_options'],
        cfg['batch_size_options'], cfg['focal_gamma_options']
    ))

    cv = MultilabelStratifiedKFold(n_splits=cfg['outer_folds'], shuffle=True, random_state=cfg['global_seed'])
    Y_train_bin = (Y_train > cfg['eval_pos_threshold']).astype(int)

    logging.info(f"Starting Training on X_train: {X_train.shape}...")

    for fold_idx, (train_ix, val_ix) in enumerate(cv.split(X_train, Y_train_bin)):
        fold_id = str(fold_idx + 1)
        X_tr, Y_tr = X_train[train_ix], Y_train[train_ix]
        X_val, Y_val = X_train[val_ix], Y_train[val_ix]

        # Inner Grid Search
        inner_cv = MultilabelStratifiedKFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['global_seed'])
        best_score = -1; best_hp = None

        for hp_t in param_grid:
            hp = {'lr': hp_t[0], 'weight_decay': hp_t[1], 'batch_size': hp_t[2], 'gamma': hp_t[3]}
            scores = []
            for i_tr, i_v in inner_cv.split(X_tr, Y_train_bin[train_ix]):
                xt, yt = X_tr[i_tr], Y_tr[i_tr]
                xv, yv = X_tr[i_v], Y_tr[i_v]

                prep = Preprocessor(cfg['pca_variance']).fit(xt)
                xt_p, xv_p = prep.transform(xt), prep.transform(xv)
                xt_aug, yt_aug = augment_train(xt_p, yt, class_list, cfg)

                y_b_aug = (yt_aug > 0.5).astype(int)
                pos_w = np.clip((y_b_aug.shape[0]-y_b_aug.sum(0))/(y_b_aug.sum(0)+1),
                                cfg['pos_weight_clamp'][0], cfg['pos_weight_clamp'][1])

                _, val_p = train_single_model(xt_aug, yt_aug, xv_p, yv, cfg, hp, pos_w)
                scores.append(f1_score((yv>0.5).astype(int), (val_p>0.5).astype(int), average='macro', zero_division=0))

            avg_s = np.mean(scores)
            if avg_s > best_score: best_score = avg_s; best_hp = hp

        # Final Fold Training
        prep = Preprocessor(cfg['pca_variance']).fit(X_tr)
        X_tr_p, X_val_p = prep.transform(X_tr), prep.transform(X_val)
        X_tr_aug, Y_tr_aug = augment_train(X_tr_p, Y_tr, class_list, cfg)

        y_b_aug = (Y_tr_aug > 0.5).astype(int)
        pos_w = np.clip((Y_tr_aug.shape[0]-y_b_aug.sum(0))/(y_b_aug.sum(0)+1),
                        cfg['pos_weight_clamp'][0], cfg['pos_weight_clamp'][1])

        model, raw_val_probs = train_single_model(X_tr_aug, Y_tr_aug, X_val_p, Y_val, cfg, best_hp, pos_w)
        calibs = fit_calibrators(raw_val_probs, Y_val, class_list, cfg)
        val_probs_cal = apply_calibrators(calibs, raw_val_probs, class_list)
        thresh = threshold_tuning(val_probs_cal, Y_val, class_list, cfg)

        oof_preds[val_ix] = val_probs_cal
        ensemble_artifacts['models'].append(deepcopy(model.state_dict()))
        ensemble_artifacts['preps'].append(prep)
        ensemble_artifacts['calibs'].append(calibs)
        ensemble_artifacts['thresh'].append(thresh)
        ensemble_artifacts['pos_weights'].append(pos_w)

    # Save and Report
    ens_model = EnsemblePredictor(ensemble_artifacts['models'], ensemble_artifacts['preps'],
                                ensemble_artifacts['calibs'], ensemble_artifacts['thresh'], device=DEVICE)
    joblib.dump(ens_model, os.path.join(ensemble_root, "ensemble_model.pkl"))
    avg_thresh = np.mean(np.array(ensemble_artifacts['thresh']), axis=0)

    compute_metrics_and_save(oof_preds, Y_train, avg_thresh, class_list,
                             os.path.join(dir_train_report, "Train_CV_Metrics.csv"), cfg)
    plot_ensemble_figures(oof_preds, Y_train, class_list, dir_train_report)
    
    test_probs = ens_model.predict_proba(X_test)
    compute_metrics_and_save(test_probs, Y_test, avg_thresh, class_list,
                             os.path.join(ensemble_root, "Ensemble_Metrics.csv"), cfg)
    plot_ensemble_figures(test_probs, Y_test, class_list, dir_test_report)
    logging.info("Pipeline Complete.")
