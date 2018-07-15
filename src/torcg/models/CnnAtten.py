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
                kernel_size=(k, vocab.embed_dim)
            ) for k in ckn
        ])

        self.dropout = nn.Dropout(dropout)
        self.le = nn.Linear(len(ckn) * cnn_bins, n_bins)

    def forward(self, input_x):
        input_emb = self.embeded(input_x)
        # x = torch.transpose(x, 0, 1)
        x = input_emb.unsqueeze(1)
        # print("==============================shape", x.size())
        # batch * 1 * sen_max * embed
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        maxpool_x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]
        # batch *  * sen_max * embe1d
        maxpool_x = torch.cat(maxpool_x, 1)
        maxpool_x = self.dropout(maxpool_x)

        r_x = self.le(maxpool_x)
        r_x = F.relu(r_x)

        r_x = self.dropout(r_x)
        return r_x


class TextCNNNP(nn.Module):
    def __init__(self, vocab, n_bins=128, cnn_bins=64, ckn=None, dropout=0.3):
        super(TextCNNNP, self).__init__()

        if ckn is None:
            ckn = [2, 3, 4, 5]
        self.embeded = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.embeded.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=cnn_bins,
                kernel_size=(k, vocab.embed_dim)
            ) for k in ckn
        ])

        self.dropout = nn.Dropout(dropout)
        self.le = nn.Linear(len(ckn) * cnn_bins, n_bins)

    def forward(self, input_x):
        input_emb = self.embeded(input_x)
        # x = torch.transpose(x, 0, 1)
        x = input_emb.unsqueeze(1)
        # print("==============================shape", x.size())
        # batch * 1 * sen_max * embed
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        maxpool_x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]
        # batch *  * sen_max * embe1d
        all_x = torch.cat(x, dim=1)
        all_x = self.dropout(all_x)

        # r_x = self.le(maxpool_x)
        r_x = F.relu(all_x)

        # r_x = self.dropout(r_x)
        return r_x


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),  # 24
            nn.ReLU(True),
            nn.Linear(24, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        attn_ene = attn_ene.view(batch_size, -1)
        return F.softmax(attn_ene, dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class CNNAttenMatch(nn.Module):
    def __init__(self, vocab, h_dim, c_num, dropout=0.3):
        super(CNNAttenMatch, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn_model = Attn(h_dim)
        self.convs = TextCNNNP(vocab, n_bins=h_dim,
                               cnn_bins=64, ckn=[1, 2, 3, 4],
                               dropout=0.5)
        self.cosout = nn.Linear(1, c_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        conv_out1 = self.convs(input1)
        conv_out2 = self.convs(input2)
        attns = self.attn_model(conv_out1)  # (b, s, 1)
        attns2 = self.attn_model(conv_out2)  # (b, s, 1)
        feats1 = (conv_out1 * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        feats2 = (conv_out2 * attns2).sum(dim=1)  # (b, s, h) -> (b, h)
        feats = F.pairwise_distance(feats1, feats2).view(input1.shape[0], 1)
        # feats = self.main(feats)
        feats = self.cosout(feats)
        feats = self.dropout(feats)
        # F.log_softmax(self.main(feats), dim=1)
        soft_outputs = F.softmax(feats, dim=-1)
        return soft_outputs, attns, attns2
