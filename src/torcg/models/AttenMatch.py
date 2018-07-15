import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class EncoderRNN(nn.Module):
    def __init__(self, vocab, h_dim, device, batch_first=True):
        super(EncoderRNN, self).__init__()
        self.h_dim = h_dim
        self.device = device
        self.embeded = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.embeded.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))
        self.lstm = nn.LSTM(vocab.embed_dim, h_dim, batch_first=batch_first, bidirectional=True)

    def init_hidden(self, b_size):
        h0 = torch.zeros(1 * 2, b_size, self.h_dim, device=self.device)
        c0 = torch.zeros(1 * 2, b_size, self.h_dim, device=self.device)
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.embeded(sentence)
        packed_emb = emb

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        out, hidden = self.lstm(packed_emb, self.hidden)

        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(out)[0]

        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

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


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num, dropout=0.3):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim)
        self.main = nn.Linear(h_dim * 2, c_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, char_encoder_outputs):
        attns = self.attn(encoder_outputs)  # (b, s, 1)
        attns2 = self.attn(char_encoder_outputs)  # (b, s, 1)
        word_feats = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        char_feats = (char_encoder_outputs * attns2).sum(dim=1)  # (b, s, h) -> (b, h)
        feats = torch.cat((word_feats, char_feats), dim=-1)
        feats = self.dropout(feats)
        return F.log_softmax(self.main(feats), dim=1), attns, attns2


class EncoderAttnClassifier(nn.Module):
    def __init__(self, vocab, h_dim, device, c_num, dropout=0.3, batch_first=True):
        super(EncoderAttnClassifier, self).__init__()
        self.encoder = EncoderRNN(vocab, h_dim, device, batch_first=batch_first)
        self.attn_model = Attn(h_dim)
        self.main = nn.Linear(h_dim, c_num)
        self.cosout = nn.Linear(1, c_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence1, sentence12):
        encoder_outputs = self.encoder(sentence1)
        encoder_outputs2 = self.encoder(sentence12)
        attns = self.attn_model(encoder_outputs)  # (b, s, 1)
        attns2 = self.attn_model(encoder_outputs2)  # (b, s, 1)
        feats1 = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        feats2 = (encoder_outputs2 * attns2).sum(dim=1)  # (b, s, h) -> (b, h)
        # feats = torch.cat((feats1, feats2), dim=-1)
        # feats = feats1 * feats2
        feats = F.pairwise_distance(feats1, feats2).view(sentence1.shape[0], 1)
        # feats = self.main(feats)
        feats = self.cosout(feats)
        feats = self.dropout(feats)
        # F.log_softmax(self.main(feats), dim=1)
        soft_outputs = F.softmax(feats, dim=-1)
        return soft_outputs, attns, attns2


class EncoderAttnSimer(nn.Module):
    def __init__(self, vocab, h_dim, device, dropout=0.3, batch_first=True):
        super(EncoderAttnSimer, self).__init__()
        self.encoder = EncoderRNN(vocab, h_dim, device, batch_first=batch_first)
        self.attn_model = Attn(h_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence1, sentence12):
        encoder_outputs = self.encoder(sentence1)
        encoder_outputs2 = self.encoder(sentence12)
        attns = self.attn_model(encoder_outputs)  # (b, s, 1)
        attns2 = self.attn_model(encoder_outputs2)  # (b, s, 1)
        feats1 = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        feats2 = (encoder_outputs2 * attns2).sum(dim=1)  # (b, s, h) -> (b, h)

        return feats1, feats2, attns, attns2


