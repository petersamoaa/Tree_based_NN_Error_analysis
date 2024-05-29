import torch
from torch import nn

from .attention import MultiHeadedCombination
from .layers import SublayerConnection
from .utils import GELU


class GCNN(nn.Module):
    def __init__(self, dmodel):
        super(GCNN, self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)

    def forward(self, state, left, inputad):
        # print(state.size(), left.size())
        state = torch.cat([left, state], dim=1)
        state = self.linear(state)
        degree = torch.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree2 = torch.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)

        degree = 1.0 / torch.sqrt(degree)
        degree2 = 1.0 / torch.sqrt(degree2)
        # print(degree2.size(), state.size())
        degree2 = degree2 * inputad * degree
        # tmp = torch.matmul(degree2, state)
        state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.matmul(degree2, state)))  # state + torch.matmul(degree2, state)
        state = self.linearSecond(state)
        return state[:, 50:, :]  # self.dropout(state)[:,50:,:]


class GCNNM(nn.Module):
    def __init__(self, dmodel):
        super(GCNNM, self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)
        self.comb = MultiHeadedCombination(8, dmodel)
        self.subconnect1 = SublayerConnection(dmodel, 0.1)

    def forward(self, state, inputad, rule):
        # print(rule.size())
        state = self.subconnect1(state, lambda _x: self.comb(_x, _x, rule, batch_size=1))  #
        state = self.linear(state)
        # print(state.size())
        degree = torch.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree2 = torch.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)

        degree = 1.0 / torch.sqrt(degree)
        degree2 = 1.0 / torch.sqrt(degree2)
        degree2 = degree2 * inputad * degree
        state2 = torch.matmul(degree2, state)
        # state = self.linearSecond(state)
        state = self.subconnect(state, lambda _x: self.com(_x, _x, state2, batch_size=1))  # state + torch.matmul(degree2, state)
        return state  # self.dropout(state)[:,50:,:]
