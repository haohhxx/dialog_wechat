# -*- coding:utf8 -*-

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

from .s2s_vocab import Vocab


class DialogPairDataSet(Dataset):

    def __init__(self, srcs, targets, vocabs, max_length=100):
        self.srcs = srcs
        self.targets = targets
        self.vocab = vocabs
        self.max_length = max_length

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, index):
        eos_token_id = self.vocab.term2id[self.vocab.eos_term]
        sos_token_id = self.vocab.term2id[self.vocab.sos_term]
        max_length = self.max_length

        def pad_que(que_line):
            que_line = sos_token_id + que_line[:max_length-2] + eos_token_id

            que_line_pad = np.zeros(max_length, dtype=np.int64)
            for i, tid in enumerate(np.asarray(que_line, dtype=np.int64)):
                que_line_pad[i] = tid
            return que_line_pad

        src = self.srcs[index]
        target = self.targets[index]
        que1 = self.vocab.convert_to_ids(src)
        que2 = self.vocab.convert_to_ids(target)

        src = pad_que(que1)
        target = pad_que(que2)

        return src, target


class DialogPairData:

    def __init__(self, train_file, pre_train_embedding, batch_size=32, max_length=100, line_nub=None):
        vocab = Vocab(lower=True)

        def add2vocab(ls):
            for l in ls:
                for word in l.split():
                    vocab.add(word)

        def load_file(file_reader):
            # lines = csv.reader(csv_file)
            lines = file_reader.readlines()[:line_nub]
            src = []
            target = []
            for line in lines:
                ls = line.split("\t")
                add2vocab(ls)
                for did in range(1, len(ls)):
                    src.append(ls[did-1])
                    target.append(ls[did])
            return src, target

        with open(train_file, 'r', encoding='utf-8') as csv_file:
            src, target = load_file(csv_file)

        # 根据不同的损失函数修改target的格式
        # target = np.array(target, dtype=np.float32)

        train_src, test_src, train_target, test_target = train_test_split(src, target, shuffle=True, test_size=0.25)

        # 构建数据  根据pair构建
        train_pair_dataset = DialogPairDataSet(train_src, train_target, vocab, max_length)
        test_pair_dataset = DialogPairDataSet(test_src, test_target, vocab, max_length)

        train_dataloader = DataLoader(train_pair_dataset, batch_size=batch_size, num_workers=6, shuffle=True)
        test_dataloader = DataLoader(test_pair_dataset, batch_size=batch_size, num_workers=6, shuffle=True)

        vocab.filter_terms_by_cnt(min_count=5)
        if pre_train_embedding is None:
            vocab.randomly_init_embeddings(300)
        else:
            vocab.load_pretrained_embeddings(pre_train_embedding)
        # vocab.randomly_init_embeddings(300)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vocab = vocab


