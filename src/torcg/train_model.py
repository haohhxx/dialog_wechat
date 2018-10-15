"""
TextCNN 对两组文本进行扫描

使用预选距离比较扫描结果
"""
import os

import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.torcg.pair_data import DialogPairData

log_dir = "../logs"
data_load_catch_path = r"./dialog_data.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=log_dir, comment='Dialog_Log')
batch_size = 32
max_length = 50


def prepare():
    # train_file = r"/mnt/haohhxx/d/data/atec/atec_nlp_sim_train_add.hanlpcut.csv"
    # pre_train_embedding = r"/mnt/haohhxx/d/data/word2vec/zh/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"
    # pre_train_embedding = r"/mnt/haohhxx/d/data/word2vec/zh/sgns.merge.word/sgns.merge.word.utf8.txt"
    train_file = r"D:/data/atec/atec_nlp_sim_train_add.hanlpcut.csv"
    pre_train_embedding = r"D:/data/word2vec/zh/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.context" \
                          r".word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt "
    dialog_data = DialogPairData(train_file, pre_train_embedding
                                 , batch_size=batch_size, max_length=max_length
                                 , line_nub=-1)
    pickle.dump(dialog_data, open(data_load_catch_path, 'wb'))
    print("catch data save at " + data_load_catch_path)


def train_eatch(train_batchs, optimizer, sim_model, riterion, epoch_i):
    all_step_nub = epoch_i * train_batchs.__len__()
    total_loss = []

    # riterion = nn.CosineEmbeddingLoss()

    # weight = torch.Tensor([1, 5]).to(device)
    # riterion = nn.CrossEntropyLoss(weight=weight)

    for batch_i, train_batch in enumerate(train_batchs):

        optimizer.zero_grad()

        que1_array, que2_array, target = train_batch

        outputs = sim_model(que1_array.to(device), que2_array.to(device))
        # BCEWithLogitsLoss 时需要求和 合并每一条的损失
        # batch_loss = riterion(q1_outputs, q2_outputs, target.cuda()).sum()
        # backward
        # print(outputs)
        batch_loss = riterion(outputs, target.to(device)).sum()

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


def valid(valid_batchs, sim_model):
    pres = []
    labels = []
    for i, valid_batch in enumerate(valid_batchs):
        que1_array, que2_array, target = valid_batch

        outputs = sim_model(que1_array.to(device), que2_array.to(device))
        _, outputs = torch.max(outputs, 1)

        pres += outputs.data.cpu().numpy().tolist()
        labels += target.data.cpu().numpy().tolist()
        # print(predict)

    f1score = f1_score(labels, pres)
    accuracy = accuracy_score(labels, pres)
    report = classification_report(labels, pres)
    return f1score, accuracy, report


def predict(output):
    _, predict = torch.max(output, 1)


def train():
    """:data"""
    atec_data = pickle.load(open(qdd, 'rb'))
    train_batchs = atec_data.train_dataloader
    valid_batchs = atec_data.test_dataloader
    vocab = atec_data.vocab

    """:parameter"""
    epoch = 100
    c_num = 2
    # learning_rate = 0.0001 TextCnnSim
    learning_rate = 0.0001
    h_dim = 32

    # sim_model = RnnSim(vocab, h_dim, device, c_num).to(device)
    sim_model = TextCnnSim(vocab, output_bins=c_num, n_bins=h_dim,
                           cnn_bins=h_dim, ckn=[1, 2, 3], dropout=0.3).to(device)
    # sim_model = Cmatch(vocab, h_dim, nlayers=1, c_nub=c_num).to(device)
    print(sim_model)

    sim_model_optim = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)

    max_test_acc = 0.0
    #    保存
    for epoch_i in range(epoch):
        total_loss = train_eatch(train_batchs,
                                 sim_model_optim,
                                 sim_model,
                                 epoch_i)
        f1score, accuracy, report = valid(valid_batchs, sim_model)

        """log"""
        writer.add_scalars('toge', {'train_loss': float(total_loss),
                                    'valid_f1score': f1score,
                                    'valid_accuracy': accuracy}, epoch_i)
        print("epoch:{} loss:{}".format(epoch_i, total_loss))
        print("epoch:{} f1score:{}".format(epoch_i, f1score))
        print("epoch:{} accuracy:{}".format(epoch_i, accuracy))
        print(report)

        # checkpoint = {
        #     'model': knrm_model,
        #     'epoch': epoch_i,
        # }
        # torch.save(checkpoint,
        #            'acc_{}_f1_{}.pt'
        #            .format(accuracy,f1score, epoch_i))


if __name__ == "__main__":
    # os.remove(log_dir)
    # os.removedirs(log_dir)
    # prepare()
    train()
