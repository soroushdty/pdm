import torch
import torch.nn as nn

def compute_focal_bce_loss(logits, targets, pos_weight, gamma=2.0):
    """Calculates Focal Loss combined with Binary Cross Entropy."""
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = bce(logits, targets)
    probs = torch.sigmoid(logits)
    # Calculate probability of the target class
    p_t = probs * targets + (1 - probs) * (1 - targets)
    return ((1 - p_t) ** gamma * loss).mean()