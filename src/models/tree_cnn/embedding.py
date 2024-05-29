"""
A script for training a neural network on tree-structured data.
"""

import torch
import torch.nn as nn


class TreeCNNEmbedding(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, num_classes, num_feats=100, hidden_size=100):
        super(TreeCNNEmbedding, self).__init__()

        # Embedding layer
        self.embeddings = nn.Embedding(num_classes, num_feats)

        # Hidden layer
        self.hidden = nn.Linear(num_feats, hidden_size)

        # Softmax layer
        self.softmax = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embeddings(x)

        # Hidden layer with tanh activation
        x = torch.tanh(self.hidden(x))

        # Logits
        logits = self.softmax(x)

        return logits
