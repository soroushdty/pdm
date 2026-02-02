import numpy as np

def apply_calibrators(calibrators, probs, class_list):
    res = np.zeros_like(probs)
    for i, cls in enumerate(class_list):
        cal = calibrators[cls]
        p_c = probs[:, i].reshape(-1,1)
        if hasattr(cal, 'predict_proba'):
            # Predict probability for the positive class (index 1)
            res[:, i] = cal.predict_proba(p_c)[:, 1]
        else:
            res[:, i] = cal.predict(p_c)
    return res