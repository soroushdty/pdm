import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             accuracy_score, f1_score, average_precision_score, 
                             roc_auc_score, precision_recall_curve, auc)

def compute_metrics_and_save(probs, y_true, thresholds, class_list, save_path, cfg):
    y_true_bin = (y_true > cfg['eval_pos_threshold']).astype(int)
    metrics_rows = []

    for i, cls in enumerate(class_list):
        t = thresholds[i]
        p_c = (probs[:, i] >= t).astype(int)
        y_c = y_true_bin[:, i]

        tn, fp, fn, tp = confusion_matrix(y_c, p_c, labels=[0, 1]).ravel()

        prec = precision_score(y_c, p_c, zero_division=0)
        rec = recall_score(y_c, p_c, zero_division=0)
        acc = accuracy_score(y_c, p_c)
        f1 = f1_score(y_c, p_c, zero_division=0)
        ap = average_precision_score(y_c, probs[:, i]) if len(np.unique(y_c)) > 1 else 0.0

        try:
            auc_roc = roc_auc_score(y_c, probs[:, i]) if len(np.unique(y_c)) > 1 else 0.5
        except: auc_roc = 0.5

        if len(np.unique(y_c)) > 1:
            p_curve, r_curve, _ = precision_recall_curve(y_c, probs[:, i])
            auc_pr = auc(r_curve, p_curve)
        else:
            auc_pr = 0.0

        metrics_rows.append({
            "Class": cls,
            "Precision": prec, "Recall": rec, "Accuracy": acc,
            "AUC ROC": auc_roc, "AUC PR": auc_pr, "AP": ap, "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "Threshold": t
        }) # Fixed logic: Assigned fp to FP and tn to TN

    df = pd.DataFrame(metrics_rows)
    numeric_cols = df.columns.drop('Class')
    means = df[numeric_cols].mean()
    macro_row = means.to_dict()
    macro_row['Class'] = 'Macro Average'
    df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)

    df.to_csv(save_path, index=False)
    logging.info(f"Saved metrics to {save_path}")
    return df