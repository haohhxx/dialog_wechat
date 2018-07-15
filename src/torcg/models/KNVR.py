"""
model class
KernelPooling: the kernel pooling layer
KNRM: base class of KNRM, can choose to:
    learn distance metric
    learn entity attention
"""
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np


class KNRM(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, vocab, n_bins, usecuda=True):
        def kernal_mus(n_kernels):
            """
            get the mu for each guassian kernel. Mu is the middle of each bin
            :param n_kernels: number of kernels (including exact match). first one is exact match
            :return: l_mu, a list of mu.
            """
            l_mu = [1]
            if n_kernels == 1:
                return l_mu

            bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
            l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
            for i in range(1, n_kernels - 1):
                l_mu.append(l_mu[i] - bin_size)
            return l_mu

        def kernel_sigmas(n_kernels):
            """
            get sigmas for each guassian kernel.
            :param n_kernels: number of kernels (including exactmath.)
            :param lamb:
            :param use_exact:
            :return: l_sigma, a list of simga
            """
            bin_size = 2.0 / (n_kernels - 1)
            l_sigma = [0.001]  # for exact match. small variance -> exact match
            if n_kernels == 1:
                return l_sigma

            l_sigma += [0.1] * (n_kernels - 1)
            return l_sigma

        super(KNRM, self).__init__()

        self.word_emb = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.word_emb.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))

        tensor_mu = torch.FloatTensor(kernal_mus(n_bins))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(n_bins))
        if usecuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, n_bins)
        self.dense = nn.Linear(n_bins, 2, 1)
        self.dropout = nn.Dropout(0.5)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2))\
            .view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        # RBF Kernel
        # mask those non-existing words.
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d

        # Kernel Pooling
        pooling_sum = torch.sum(pooling_value, 2)

        # Soft-TF Features
        # clamp截断最大最小值
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed = self.word_emb(inputs_q)
        d_embed = self.word_emb(inputs_d)
        # 归一化？
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        # squeeze 将张量中只有1的维度去掉
        # output = torch.squeeze(F.tanh(self.dense(log_pooling_sum)), 1)
        # output = torch.squeeze(F.log_softmax(self.dense(log_pooling_sum), dim=-1), 1)
        lpo = self.dense(log_pooling_sum)
        # lpo = self.dropout(lpo)
        output = torch.squeeze(F.softmax(lpo, dim=-1), 1)
        # output = torch.squeeze(F.log_softmax(lpo, dim=-1), 1)
        return output
