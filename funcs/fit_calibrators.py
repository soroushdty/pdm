import numpy as np
from sklearn.linear_model import LogisticRegression

# Placeholder for ConstantCalibrator if not imported from elsewhere
class ConstantCalibrator:
    def __init__(self, val): self.val = val
    def predict(self, X): return np.full((X.shape[0],), self.val)

def fit_calibrators(probs, Y_true, class_list, cfg):
    calibrators = {}
    Y_bin = (Y_true > cfg['eval_pos_threshold']).astype(int)
    for i, cls in enumerate(class_list):
        if len(np.unique(Y_bin[:, i])) < 2:
            calibrators[cls] = ConstantCalibrator(Y_bin[:, i].mean())
        else:
            clf = LogisticRegression(C=0.1, solver='lbfgs')
            clf.fit(probs[:, i].reshape(-1, 1), Y_bin[:, i])
            calibrators[cls] = clf
    return calibrators