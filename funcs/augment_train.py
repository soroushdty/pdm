import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from .set_seeds import set_seeds 

def augment_train(X_train, Y_train, class_list, cfg):
    set_seeds(cfg['global_seed'])
    N = X_train.shape[0]
    cutoff = int(cfg['smote_rarity_cutoff'] * N)
    X_aug, Y_aug = [X_train], [Y_train]
    Y_bin = (Y_train > cfg['smote_pos_threshold']).astype(int)
    counts = Y_bin.sum(axis=0)

    rare_indices = [i for i, c in enumerate(counts) if c < cutoff]

    for c_idx in rare_indices:
        pos_mask = Y_bin[:, c_idx] == 1
        pos_ix = np.where(pos_mask)[0]
        if len(pos_ix) == 0: continue
        n_needed = max(0, cutoff - len(pos_ix))
        if n_needed == 0: continue

        if len(pos_ix) < cfg.get('smote_min_samples_for_smote', 6):
            choices = np.random.choice(pos_ix, size=n_needed, replace=True)
            X_aug.append(X_train[choices])
            Y_aug.append(Y_train[choices])
        else:
            try:
                sm = SMOTE(sampling_strategy={1: len(pos_ix) + n_needed}, 
                           k_neighbors=min(len(pos_ix)-1, 5), 
                           random_state=cfg['global_seed'])
                y_tmp = np.zeros(N, dtype=int); y_tmp[pos_ix] = 1
                X_res, _ = sm.fit_resample(X_train, y_tmp)
                X_syn = X_res[N:]
                if len(X_syn) > 0:
                    nn = NearestNeighbors(n_neighbors=1).fit(X_train[pos_ix])
                    _, neighbors = nn.kneighbors(X_syn)
                    Y_syn = Y_train[pos_ix[neighbors.flatten()]].copy()
                    Y_syn[:, c_idx] = np.maximum(Y_syn[:, c_idx], 0.80)
                    X_aug.append(X_syn); Y_aug.append(Y_syn)
            except:
                 choices = np.random.choice(pos_ix, size=n_needed, replace=True)
                 X_aug.append(X_train[choices])
                 Y_aug.append(Y_train[choices])

    return np.vstack(X_aug), np.vstack(Y_aug)
