import numpy as np
import torch
import random

def set_seeds(seed):
    """Sets seeds for reproducibility across numpy, random, and torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True