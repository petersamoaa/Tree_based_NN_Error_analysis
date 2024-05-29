import torch.nn as nn
import torch.nn.functional as F


class Code2VecConvNet(nn.Module):
    def __init__(self, embedding_dim, label_size=1):
        super(Code2VecConvNet, self).__init__()

        # Fully connected layers
        self.fc = nn.Linear(embedding_dim, label_size)

    def forward(self, x):
        # Input x: N x embedding_dim
        # Fully connected layers
        x = F.relu(self.fc(x))
        return x
