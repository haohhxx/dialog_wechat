"""
model class
"""
import torch.nn.functional as F
import torch
import torch.nn as nn


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class EncoderRNN(nn.Module):
    def __init__(self, vocab, h_dim, device):
        super(EncoderRNN, self).__init__()
        self.h_dim = h_dim
        self.device = device
        self.embeded = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.embeded.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))
        self.lstm = nn.LSTM(vocab.embed_dim, h_dim, batch_first=False, bidirectional=True, bias=True,)

    def init_hidden(self, b_size):
        h0 = torch.zeros(1 * 2, b_size, self.h_dim, device=self.device)
        c0 = torch.zeros(1 * 2, b_size, self.h_dim, device=self.device)
        return (h0, c0)

    def forward(self, sentence):
        emb = self.embeded(sentence)
        packed_emb = emb.permute(1, 0, 2)
        out = self.lstm(packed_emb)[0].permute(1, 2, 0)  # batch max encode
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]  # 前后段求和
        return out


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, h_dim),  # 24
            nn.ReLU(True),
            nn.Linear(h_dim, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.view(-1, self.h_dim)
        attn_ene = self.main(encoder_outputs)  # (b, s, h) -> (b * s, 1)
        attn_ene = attn_ene.view(b_size, -1)
        return F.softmax(attn_ene, dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class RnnSim(nn.Module):

    def __init__(self, vocab, hidden_size, device, c_nub, kmax_pooling=32, dropout=0.3):
        super(RnnSim, self).__init__()
        self.kmax_pooling = kmax_pooling
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.encoder = EncoderRNN(vocab, hidden_size, device)
        self.attn_model = Attn(hidden_size * 2)
        self.device = device
        self.out = nn.Linear(1, c_nub)

        self.fc = nn.Sequential(
            nn.Linear(kmax_pooling * (hidden_size * 2), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, input_q1, input_q2):
        encoder_out_q1 = self.encoder(input_q1)
        encoder_out_q2 = self.encoder(input_q2)
        # max_out_q1 = F.max_pool1d(encoder_out_q1, kernel_size=encoder_out_q1.size(2))
        # max_out_q2 = F.max_pool1d(encoder_out_q2, kernel_size=encoder_out_q2.size(2))
        atin1 = encoder_out_q1.permute(0, 2, 1).contiguous()
        atin2 = encoder_out_q2.permute(0, 2, 1).contiguous()
        attns_q1 = self.attn_model(atin1)
        attns_q2 = self.attn_model(atin2)
        max_out_q1 = (atin1 * attns_q1).permute(0, 2, 1).contiguous()  # (b, s, h) -> (b, h)
        max_out_q2 = (atin2 * attns_q2).permute(0, 2, 1).contiguous()  # (b, s, h) -> (b, h)

        max_out_q1 = kmax_pooling(max_out_q1, dim=2, k=self.kmax_pooling)
        max_out_q2 = kmax_pooling(max_out_q2, dim=2, k=self.kmax_pooling)

        # outputs = pout.view(input_q1.shape[0], 1)
        # outputs = F.pairwise_distance(q1_out, q2_out).view(input_q1.shape[0], 1)
        max_out_q1 = max_out_q1.view(max_out_q1.size(0), -1)
        max_out_q2 = max_out_q2.view(max_out_q2.size(0), -1)
        fc_max_out_q1 = self.fc(max_out_q1)
        fc_max_out_q2 = self.fc(max_out_q2)
        # batch * maxlen * hidden
        feats = F.pairwise_distance(fc_max_out_q1, fc_max_out_q2).view(input_q1.shape[0], 1)

        # outputs = torch.cat((decoder_output_q1, decoder_output_q2), dim=1)
        # batch * maxlen * hidden * 2
        # outputs = self.conv(outputs)
        outputs = self.out(feats)
        outputs = self.dropout(outputs)
        soft_outputs = F.softmax(outputs, dim=-1)
        return soft_outputs


