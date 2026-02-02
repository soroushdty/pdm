import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression


# Get the absolute path of the directory above the current working directory
module_path = os.path.abspath(os.path.join('..'))
# Insert the path into sys.path at index 0 to ensure it is checked first
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from cls.ConstantCalibrator import ConstantCalibrator


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
