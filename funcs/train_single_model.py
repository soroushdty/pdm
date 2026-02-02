import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score
from .compute_focal_bce_loss import compute_focal_bce_loss
from .set_seeds import set_seeds

# Get the absolute path of the directory above the current working directory
module_path = os.path.abspath(os.path.join('..'))
# Insert the path into sys.path at index 0 to ensure it is checked first
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from cls.MultiLabelLogistic import MultiLabelLogistic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_single_model(X_train, Y_train, X_val, Y_val, cfg, hp, pos_weight_vector):
    set_seeds(cfg['global_seed'])
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    pos_w = torch.tensor(pos_weight_vector, dtype=torch.float32).to(DEVICE)

    model = MultiLabelLogistic(X_train.shape[1], Y_train.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    best_f1 = -1.0; best_state = None; patience = 0
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=hp['batch_size'], shuffle=True)

    for epoch in range(cfg['num_epochs']):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = compute_focal_bce_loss(model(xb), yb, pos_w, hp['gamma'])
            loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            v_probs = torch.sigmoid(model(X_v)).cpu().numpy()

        Y_v_bin = (Y_val > cfg['eval_pos_threshold']).astype(int)
        v_preds = (v_probs > cfg['eval_pos_threshold']).astype(int)
        val_f1 = f1_score(Y_v_bin, v_preds, average='macro', zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1; best_state = deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
        if patience >= cfg['early_stopping_patience']: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_v)).cpu().numpy()
    return model, final_probs
