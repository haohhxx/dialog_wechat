"""
model class
"""
import torch.nn.functional as F
import torch
import torch.nn as nn


class EncoderRNNFeed(nn.Module):
    def __init__(self, vocab, hidden_size, max_length, nub_layzers=1):
        super(EncoderRNNFeed, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embed_dim = vocab.embed_dim
        self.embedding = nn.Embedding(vocab.size(), self.embed_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))

        self.rnn = nn.GRU(self.embed_dim, self.hidden_size, nub_layzers, batch_first=True)
        # 输入尺寸 输出尺寸 层数
        # self.gru = nn.RNN(hidden_size, hidden_size, nub_layzers, batch_first=True)

    def forward(self, input, hidden, batch_size):
        embedded = self.embedding(input)
        encoder_output = torch.zeros(batch_size, 1, self.hidden_size, device='cuda')
        # encoder_output = torch.zeros(64, self.max_length, self.hidden_size, device='cuda')

        for i, emb_t in enumerate(embedded.split(1, dim=1)):
            output_t, hidden = self.rnn(emb_t, hidden)
            # index = torch.LongTensor(i).cuda()
            # encoder_output = encoder_output.scatter_(1, index, output_t)
            # index_fill_
            # index = torch.LongTensor(i).cuda()
            # scatter_

            encoder_output = torch.cat((encoder_output, output_t), dim=1)
            # encoder_output[-1, i] = output_t
            # encoder_output
        encoder_output = encoder_output[:, 1:]
        return encoder_output, hidden


class DecoderRNNFeed(nn.Module):
    def __init__(self, tgt_vocab, hidden_size, output_size=100,
                 nub_layzers=1, dropout_p=0.1, max_length=100):
        super(DecoderRNNFeed, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embed_dim = tgt_vocab.embed_dim

        self.embedding = nn.Embedding(tgt_vocab.size(), self.embed_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(tgt_vocab.embeddings))

        self.attn = nn.Linear(self.hidden_size + self.embed_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.embed_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # 输入尺寸 输出尺寸 层数
        self.rnn = nn.GRU(self.hidden_size, self.output_size, nub_layzers, batch_first=True)

    def forward(self, tgt_input, hidden, encoder_outputs, batch_size):
        embedded = self.embedding(tgt_input)
        embedded = self.dropout(embedded)

        # decoder_attentions = torch.zeros(self.tgt_input.shape(0), self.max_length, self.max_length)
        # output = torch.zeros(self.tgt_input.shape(0), self.max_length, self.max_length)

        decoder_attentions = torch.zeros(batch_size, 1, self.hidden_size, device='cuda')
        decoder_outputs = torch.zeros(batch_size, 1, self.hidden_size, device='cuda')

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(embedded.split(1, dim=1)):    # batch  *  1 * emb_len
            # emb_t = emb_t.squeeze(0)

            cat_atin = torch.cat((emb_t[:, 0, :], hidden[0, :, :]), dim=1)  # batch * (max_length + emb_len)
            # cat_atin = torch.unsqueeze(cat_atin, dim=0)
            attn_weight_t = F.softmax(self.attn(cat_atin), dim=1)   # batch * max_len
            attn_weight_t = torch.unsqueeze(attn_weight_t, dim=1)   # batch * 1 * max_len
            # 相乘 encoder_outputs batch * max_len * hidden
            attn_applied = torch.bmm(attn_weight_t, encoder_outputs)

            # 拼接
            output_t = torch.cat((emb_t[:, 0, :], attn_applied[:, 0, :]), dim=1)
            # batch * (max_length + emb_len)
            output_t = self.attn_combine(output_t).unsqueeze(1)
            # batch * hidden
            output_t = F.relu(output_t)
            output_t, hidden = self.rnn(output_t, hidden)
            # 64 * 1 * 100
            # atten 64 * 1 * 100
            # decoder_attentions[i] = attn_weight_t.data
            decoder_attentions = torch.cat((decoder_attentions, attn_weight_t), dim=1)
            decoder_outputs = torch.cat((decoder_outputs, attn_weight_t), dim=1)

        decoder_outputs = decoder_outputs[:, 1:]
        decoder_attentions = decoder_attentions[:, 1:]
        return decoder_outputs, hidden, decoder_attentions


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

