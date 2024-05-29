import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Code2vecNet(nn.Module):
    def __init__(self, feature_size=384, label_size=1):
        super().__init__()

        self.feature_size = feature_size
        self.label_size = label_size

        # Latent layers
        self.h1 = nn.Linear(feature_size, feature_size)
        self.h2 = nn.Linear(feature_size, feature_size)

        # Hidden layer
        # self.output = nn.Linear(feature_size, label_size)
        self.regression_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, label_size)
        )

    def pooling_layer(self, x):
        return torch.mean(x, dim=1)

    def forward(self, x):
        # x: batch_size x number of path of ast in our list x embedding of code2vec
        x = self.h1(x) 
        x = self.pooling_layer(x)
        x = self.h2(x)
        # output = self.output(x)
        outputs = self.regression_head(x)
        return outputs

class Code2vecNetV2(nn.Module):
    def __init__(self, feature_size=384, label_size=1, out_channels=128, kernel_size=3, padding=1, stride=1, pool_kernel_size=2, pool_stride=2):
        super().__init__()

        # Convolutional layer
        self.conv1 = nn.Conv1d(feature_size, out_channels, kernel_size, stride, padding)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # Additional layers can be added here for deeper networks
        
        # Adaptive pooling layer to ensure a fixed size output irrespective of the input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(out_channels, label_size)

    def forward(self, x):
        # x expected to be of shape [batch_size, feature_size, N] for Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)  # This ensures the output is of fixed size
        x = torch.flatten(x, 1)
        output = self.fc(x)
        return output