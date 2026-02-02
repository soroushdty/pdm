import torch.nn as nn
class MultiLabelLogistic(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MultiLabelLogistic, self).__init__()
        self.linear = nn.Linear(n_features, n_classes)
    def forward(self, x):
        return self.linear(x)
