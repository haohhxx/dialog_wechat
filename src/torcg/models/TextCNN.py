"""
model class
"""
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, vocab, n_bins=128, cnn_bins=64, ckn=None, dropout=0.3):
        super(TextCNN, self).__init__()

        if ckn is None:
            ckn = [2, 3, 4, 5]
        self.embeded = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.embeded.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=cnn_bins,
                kernel_size=(k, vocab.embed_dim),
                stride=1
            )for k in ckn
        ])

        self.dropout = nn.Dropout(dropout)
        self.le = nn.Linear(len(ckn)*cnn_bins, n_bins)

    def forward(self, input_x):
        input_emb = self.embeded(input_x)
        input_emb = self.dropout(input_emb)
        # x = torch.transpose(x, 0, 1)
        x = input_emb.unsqueeze(1)
        # print("==============================shape", x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        # x = self.dropout(x)
        x = self.le(x)

        x = F.relu(x)
        x = self.dropout(x)

        x = F.softmax(x, dim=-1)
        return x


class TextCnnSim(nn.Module):
    def __init__(self, vocab, output_bins=2, n_bins=128, cnn_bins=64, ckn=None, dropout=0.3):
        super(TextCnnSim, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.cnn_module = TextCNN(vocab, n_bins=n_bins, cnn_bins=cnn_bins, ckn=ckn, dropout=dropout)
        self.out_layer = nn.Linear(1, output_bins)

    def forward(self, input_q1, input_q2):
        q1_out = self.cnn_module(input_q1)
        q2_out = self.cnn_module(input_q2)
        # pout = q1_out * q2_out
        # outputs = pout.view(input_q1.shape[0], 1)
        outputs = F.pairwise_distance(q1_out, q2_out).view(input_q1.shape[0], 1)
        outputs = self.out_layer(outputs)
        outputs = self.dropout(outputs)
        soft_outputs = F.softmax(outputs, dim=-1)
        return soft_outputs

