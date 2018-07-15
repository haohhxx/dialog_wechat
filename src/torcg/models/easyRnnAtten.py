"""
model class
"""
import torch.nn.functional as F
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextCNNMatch(nn.Module):
    def __init__(self, hidden_size, output_bins=2, ckn=None, dropout=0.3):
        super(TextCNNMatch, self).__init__()

        if ckn is None:
            ckn = [1, 2, 3]
        self.output_bins = output_bins
        self.convs = nn.ModuleList([
            nn.Conv2d(
                1,
                hidden_size,  # 输出
                (k, hidden_size)  # 输入
            )for k in ckn
        ])

        self.dropout = nn.Dropout(dropout)
        self.le = nn.Linear(len(ckn)*hidden_size, hidden_size)
        self.le2 = nn.Linear(hidden_size, output_bins)

    def forward(self, input_x):
        # input_emb = self.embeded(input_x)
        # x = torch.transpose(x, 0, 1)
        input_x = input_x.unsqueeze(1)
        # print("==============================shape", x.size())
        x = [F.relu(conv(input_x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.le(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.le2(x)

        x = F.softmax(x, dim=-1)
        return x


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


class RnnCnnPo(nn.Module):
    def __init__(self, vocab, hidden_size, dropout=0.3, max_length=100):

        super(RnnCnnPo, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.encoder = EncoderRNNFeed(vocab, hidden_size, max_length=max_length)
        self.decoder = DecoderRNNFeed(vocab, hidden_size, output_size=100, max_length=max_length)
        # self.out_layer = nn.Linear(hidden_size * 2, 2)
        self.conv = TextCNNMatch(hidden_size + max_length)

    def forward(self, input_q1, input_q2):
        batch_size = input_q1.shape[0]
        hidden_q1 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        hidden_q2 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        encoder_out_q1, out_hidden_q1 = self.encoder(input_q1, hidden_q1, batch_size)
        encoder_out_q2, out_hidden_q2 = self.encoder(input_q2, hidden_q2, batch_size)

        encoder_output_q1, dncoder_hidden, decoder_attentions_q1 = \
            self.decoder(input_q2, out_hidden_q1, encoder_out_q1, batch_size)
        encoder_output_q2, dncoder_hidden, decoder_attentions_q2 = \
            self.decoder(input_q1, out_hidden_q2, encoder_out_q2, batch_size)

        # outputs = pout.view(input_q1.shape[0], 1)
        # outputs = F.pairwise_distance(q1_out, q2_out).view(input_q1.shape[0], 1)
        # batch * maxlen * hidden

        outputs = torch.cat((encoder_output_q1, encoder_output_q2), dim=2)
        # batch * maxlen * hidden * 2
        outputs = self.conv(outputs)
        outputs = self.dropout(outputs)
        soft_outputs = F.softmax(outputs, dim=-1)
        return soft_outputs


class RnnPo(nn.Module):
    def __init__(self, vocab, hidden_size, dropout=0.3, max_length=100):

        super(RnnPo, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.encoder = EncoderRNNFeed(vocab, hidden_size, max_length=max_length)
        self.decoder = DecoderRNNFeed(vocab, hidden_size, output_size=hidden_size, max_length=max_length)

        self.out_l = nn.Linear(hidden_size, 1)
        self.out_2 = nn.Linear(1, 2)

    def forward(self, input_q1, input_q2):
        batch_size = input_q1.shape[0]
        hidden_q1 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        hidden_q2 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        encoder_out_q1, out_hidden_q1 = self.encoder(input_q1, hidden_q1, batch_size)
        encoder_out_q2, out_hidden_q2 = self.encoder(input_q2, hidden_q2, batch_size)

        decoder_output_q1, dncoder_hidden_q1, decoder_attentions_q1 = \
            self.decoder(input_q2, out_hidden_q1, encoder_out_q1, batch_size)
        decoder_output_q2, dncoder_hidden_q2, decoder_attentions_q2 = \
            self.decoder(input_q1, out_hidden_q2, encoder_out_q2, batch_size)

        # outputs = pout.view(input_q1.shape[0], 1)
        # outputs = F.pairwise_distance(q1_out, q2_out).view(input_q1.shape[0], 1)
        # batch * maxlen * hidden
        decoder_output_q1 = self.out_l(decoder_output_q1)[:, :, 0]
        decoder_output_q2 = self.out_l(decoder_output_q2)[:, :, 0]

        outputs = F.pairwise_distance(decoder_output_q1, decoder_output_q2).view(input_q1.shape[0], 1)

        # outputs = torch.cat((decoder_output_q1, decoder_output_q2), dim=1)
        # batch * maxlen * hidden * 2
        # outputs = self.conv(outputs)
        outputs = self.out_2(outputs)
        outputs = self.dropout(outputs)
        soft_outputs = F.softmax(outputs, dim=-1)
        return soft_outputs


