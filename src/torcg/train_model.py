
import os

import pickle
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.torcg.models.seq2seq import Seq2Seq
from src.torcg.pair_data import DialogPairData

log_dir = "../logs"
data_load_catch_path = r"./dialog_data.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=log_dir, comment='Dialog_Log')
batch_size = 8
max_length = 30


def prepare():
    train_file = r"C:\code\python3workspace\dialog_wechat\corpus\dialog_datas\sentence_dialog.txt"
    pre_train_embedding = None
    dialog_data = DialogPairData(train_file, pre_train_embedding
                                 , batch_size=batch_size, max_length=max_length
                                 , line_nub=None)
    pickle.dump(dialog_data, open(data_load_catch_path, 'wb'))
    print("catch data save at " + data_load_catch_path)


def train_eatch(train_batchs, optimizer, model, riterion, epoch_i):
    all_step_nub = epoch_i * train_batchs.__len__()
    total_loss = []

    for batch_i, train_batch in enumerate((train_batchs)):

        optimizer.zero_grad()

        src, target = train_batch

        decoder_outputs = model(src.to(device))

        batch_loss = 0.0
        for step, step_output in enumerate((decoder_outputs)):
            batch_size = target.size(0)
            step_target = target[:, step].to(device)
            step_loss = riterion(step_output.contiguous().view(batch_size, -1), step_target)
            batch_loss += step_loss

        batch_loss.backward()

        # update parameters
        optimizer.step()

        total_loss.append(batch_loss.data)
        if batch_i % 100 == 0:
            print(batch_i, batch_loss)
            # writer.add_scalar('batch_loss', batch_loss, batch_i)
        writer.add_scalar('train_batch_loss', batch_loss, all_step_nub + batch_i)
        # if epoch_i == 0 and batch_i == 0:
        #     writer.add_graph(knrm_model, (Variable(que1_array),
        #                                   Variable(que2_array),
        #                                   Variable(que1_array_mask),
        #                                   Variable(que2_array_mask))
        #                      )

    return np.mean(total_loss)


def valid(valid_batchs, model, riterion):

    loss = 0.0
    for i, valid_batch in enumerate((valid_batchs)):
        src, target = valid_batch

        decoder_outputs = model(src.to(device))
        batch_loss = 0.0
        for step, step_output in enumerate((decoder_outputs)):
            batch_size = target.size(0)
            step_target = target[:, step].to(device)
            step_loss = riterion(step_output.contiguous().view(batch_size, -1), step_target)
            batch_loss += step_loss

        loss += batch_loss
        # print(predict)
    return loss


def predict():
    import jieba

    s2smodel_checkpoint = torch.load(r"C:\code\python3workspace\dialog_wechat\src\torcg\epoch_0.pt")
    s2smodel = s2smodel_checkpoint["model"]
    s2svocab = s2smodel_checkpoint["vocab"]
    eos_token_id = s2svocab.term2id[s2svocab.eos_term]
    sos_token_id = s2svocab.term2id[s2svocab.sos_term]

    def pad_que(que_line):
        que_line = [sos_token_id] + que_line[:max_length - 2] + [eos_token_id]

        que_line_pad = np.zeros(max_length, dtype=np.int64)
        for i, tid in enumerate(np.asarray(que_line, dtype=np.int64)):
            que_line_pad[i] = tid
        return que_line_pad
    src = "祝 你 早日 毕业"
    # src = list(jieba.cut(src))
    src = src.split(" ")
    src = s2svocab.convert_to_ids(src)
    src = pad_que(src)
    src = torch.LongTensor(src).to(device).unsqueeze(0)
    decoder_outputs = s2smodel(src)
    for outputs in decoder_outputs:
        term_id = torch.max(outputs, 1)[1].cpu().tolist()[0]
        ow = s2svocab.id2term[term_id]
        print(ow)


def train():
    """:data"""
    dialog_data = pickle.load(open(data_load_catch_path, 'rb'))
    train_batchs = dialog_data.train_dataloader
    valid_batchs = dialog_data.test_dataloader
    vocab = dialog_data.vocab

    """:parameter"""
    epoch = 1
    # learning_rate = 0.0001 TextCnnSim
    learning_rate = 0.001
    h_dim = 32

    model = Seq2Seq(vocab, max_len=max_length, hidden_size=h_dim).to(device)
    print(model)

    sim_model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    riterion = nn.NLLLoss()
    #    保存
    for epoch_i in range(epoch):
        total_loss = train_eatch(train_batchs,
                                 sim_model_optim,
                                 model,
                                 riterion,
                                 epoch_i)
        test_loss = valid(valid_batchs, model, riterion)

        """log"""
        writer.add_scalars('toge', {'train_loss': float(total_loss),
                                    'test_loss': float(test_loss)
                                    }, epoch_i)
        print("epoch:{} loss:{}".format(epoch_i, total_loss))
        print("epoch:{} test_loss:{}".format(epoch_i, test_loss))

        checkpoint = {
            'model': model,
            'vocab': vocab,
        }
        torch.save(checkpoint, 'epoch_{}.pt'.format(epoch_i))


if __name__ == "__main__":
    prepare()
    train()
    predict()
