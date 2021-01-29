#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense, norm

from src.method.layers import DeepGCNLayer

t.manual_seed(31)


class Model(nn.Module):
    def __init__(self, n_protein, n_term, n_view, dim=500, alpha=0.5, theta=0.5, dropout=0.5):
        super(Model, self).__init__()

        self.n_protein = n_protein
        self.n_term = n_term
        self.n_view = n_view

        self.bn_in = nn.ModuleList([norm.BatchNorm(n_protein)
                                    for _ in range(n_view)])
        self.linear_in = nn.ModuleList([nn.Linear(n_protein, dim, bias=True)
                                        for _ in range(n_view)])
        self.gcn1 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=1, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn2 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=2, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn3 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=3, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn4 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=4, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn5 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=5, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn6 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=6, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn7 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=7, dropout=dropout)
                                   for _ in range(n_view)])
        self.gcn8 = nn.ModuleList([DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=8, dropout=dropout)
                                   for _ in range(n_view)])
        self.linear_concat = nn.Linear(dim * n_view, dim)
        self.dropout_concat = nn.Dropout(p=dropout)
        self.classify = nn.Linear(dim, n_term)

    def forward(self, x, net):
        x = [t.squeeze(self.bn_in[i](x[i])) for i in range(self.n_view)]
        x = [F.leaky_relu(self.linear_in[i](x[i])) for i in range(self.n_view)]
        hidden = [self.gcn1[i](x[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn2[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn3[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn4[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn5[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn6[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn7[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = [self.gcn8[i](hidden[i], x[i], net[i]) for i in range(self.n_view)]
        hidden = t.cat(hidden, dim=1)
        emb = self.linear_concat(hidden)
        emb = F.leaky_relu(emb)
        emb = self.dropout_concat(emb)
        pred = self.classify(emb)
        return pred
