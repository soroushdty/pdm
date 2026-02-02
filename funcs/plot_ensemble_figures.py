import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_ensemble_figures(probs, y_true, class_list, base_dir):
    y_true_bin = (y_true > 0.5).astype(int)

    roc_dir = os.path.join(base_dir, "figures", "ROC")
    pr_dir = os.path.join(base_dir, "figures", "PR")
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC={auc_micro:.2f})', linestyle=':', linewidth=3, color='deeppink')

    # Updated to use modern colormap access
    colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(class_list)))
    for i, cls in enumerate(class_list):
        if len(np.unique(y_true_bin[:, i])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        auc_c = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], label=f'{cls} (AUC={auc_c:.2f})', linewidth=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(roc_dir, "ROC_Ensemble_Total.png"))
    plt.close()

    # 2. PR Curves
    plt.figure(figsize=(10, 8))
    prec_m, rec_m, _ = precision_recall_curve(y_true_bin.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_true_bin, probs, average='micro')
    plt.plot(rec_m, prec_m, label=f'Micro-average (AP={ap_micro:.2f})', linestyle=':', linewidth=3, color='navy')

    for i, cls in enumerate(class_list):
        if len(np.unique(y_true_bin[:, i])) < 2: continue
        p, r, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap_c = average_precision_score(y_true_bin[:, i], probs[:, i])
        plt.plot(r, p, color=colors[i], label=f'{cls} (AP={ap_c:.2f})', linewidth=2, alpha=0.8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Ensemble PR Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(pr_dir, "PR_Ensemble_Total.png"))
    plt.close()

    # 3. Per Class Separate Plots
    for i, cls in enumerate(class_list):
        clean_cls = str(cls).replace(' ', '_').replace('/', '_')
        if len(np.unique(y_true_bin[:, i])) < 2: continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"ROC - {cls}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(roc_dir, f"ROC_{clean_cls}.png"))
        plt.close()

        p, r, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], probs[:, i])
        plt.figure()
        plt.plot(r, p, label=f'AP={ap:.2f}')
        plt.title(f"PR - {cls}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(pr_dir, f"PR_{clean_cls}.png"))
        plt.close()