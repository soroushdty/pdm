import numpy as np
from sklearn.metrics import fbeta_score

def threshold_tuning(probs, Y_val, class_list, cfg):
    Y_bin = (Y_val > cfg['eval_pos_threshold']).astype(int)
    baselines = np.clip(Y_bin.mean(axis=0), 1e-6, 0.999)
    best_thresh = np.zeros(len(class_list))
    for i in range(len(class_list)):
        best_score = -1; best_t = 0.5
        for m in cfg['threshold_multipliers']:
            t = np.clip(baselines[i] * m, 0.001, 0.999)
            score = fbeta_score(Y_bin[:, i], (probs[:, i]>=t).astype(int), beta=0.5, zero_division=0)
            if score > best_score: best_score = score; best_t = t
        best_thresh[i] = best_t
    return best_thresh