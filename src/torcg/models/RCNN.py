import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.neighbors.dist_metrics import DistanceMetric
from torch.autograd import Variable

from CKNRM.src.models.AllDistance import AllDistance


class RCNN(nn.Module):
    def __init__(self, vocab, hidden_dim, conv_dim, out_dim, max_len, device):
        super(RCNN, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.use_gpu = torch.cuda.is_available()
        self.embedding_dim = vocab.embed_dim
        self.word_embeddings = nn.Embedding(vocab.size(), vocab.embed_dim)
        self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(vocab.embeddings))
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(self.embedding_dim, hidden_dim)

        self.conv = nn.Conv1d(
                      in_channels=hidden_dim * 2,
                      out_channels=conv_dim,
                      kernel_size=max_len,
                      stride=self.embedding_dim)

        self.hidden2label = nn.Linear(conv_dim, out_dim)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1 * 2, batch_size, self.hidden_dim, device=self.device)
        c0 = torch.zeros(1 * 2, batch_size, self.hidden_dim, device=self.device)
        return (h0, c0)
        #    @profile

    def forward(self, sentence):
        self.hidden = self.init_hidden(sentence.size(0))
        embeds = self.word_embeddings(sentence)  # 64x200x300

        #        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  ###input (seq_len, batch, input_size) #Outupts:output, (h_n, c_n) output:(seq_len, batch, hidden_size * num_directions)
        # lstm_out 200x64x128  lstm_out.permute(1,2,0):64x128x200
        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.conv(lstm_out)  ###64x256x1
        ###y = self.conv(lstm_out.permute(1,2,0).contiguous().view(self.batch_size,128,-1))
        # y  = self.hidden2label(y.view(sentence.size()[0],-1))
        y = self.hidden2label(conv_out.view(conv_out.size()[0], -1))  # 64x3
        return y


class RCNNMatch(nn.Module):
    def __init__(self, vocab, h_dim, device, c_num, max_len, dropout=0.3):
        super(RCNNMatch, self).__init__()
        self.encoder = RCNN(vocab=vocab,
                            hidden_dim=h_dim,
                            conv_dim=h_dim,
                            out_dim=int(h_dim/2),
                            max_len=max_len,
                            device=device)
        self.distance = AllDistance(device)
        self.cosout = nn.Linear(12, c_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence1, sentence12):
        encoder_outputs = self.encoder(sentence1)
        encoder_outputs2 = self.encoder(sentence12)
        # attns = self.attn_model(encoder_outputs)  # (b, s, 1)
        # attns2 = self.attn_model(encoder_outputs2)  # (b, s, 1)
        # feats1 = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        # feats2 = (encoder_outputs2 * attns2).sum(dim=1)  # (b, s, h) -> (b, h)
        # feats = torch.cat((feats1, feats2), dim=-1)
        # feats = feats1 * feats2
        # feats = F.pairwise_distance(encoder_outputs, encoder_outputs2).view(sentence1.shape[0], 1)
        # feats = self.main(feats)
        feats = self.distance(encoder_outputs, encoder_outputs2)
        feats = self.cosout(feats)
        feats = self.dropout(feats)
        # F.log_softmax(self.main(feats), dim=1)
        soft_outputs = F.softmax(feats, dim=-1)
        return soft_outputs






