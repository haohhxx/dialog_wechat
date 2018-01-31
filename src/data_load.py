#! /usr/bin/python
# -*- coding: utf8 -*-

import word2vec
import numpy as np


def load_w2vec(vec_bin_path):
    w2v = word2vec.load(vec_bin_path)
    vectors = w2v.vectors
    word2id = {}
    id2word = {}
    for i, j in enumerate(w2v.vocab):
        id2word[i] = j
        word2id[j] = i
    return word2id, id2word, vectors


class DataLoader(object):

    iternub = 1
    max_sentence_length = 0
    datas = []
    lines = []
    train_batch_index = 0

    def __init__(self, batch_size=30
                 , content_path=r'..\corpus\dialog_datas\sentence_dialog.txt'
                 , vec_bin_path=r'..\corpus\dialog_datas\sentence_vocbulart.txt.phrases.bin'
                 , max_iter=50):

        self.word_to_id, self.id_to_word, self.vectors = load_w2vec(vec_bin_path)

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.content_path = content_path
        self.statistic_voc()
        self.x_array = np.zeros([len(self.lines), self.max_sentence_length], dtype=np.int32)
        self.x_array_length = np.zeros([len(self.lines)], dtype=np.int32)
        self.y_array = np.zeros([len(self.lines), self.max_sentence_length], dtype=np.int32)
        self.y_array_length = np.zeros([len(self.lines)], dtype=np.int32)
        self.train_test_index = int(len(self.lines)/2)
        self.test_batch_index = self.train_test_index
        self.load()

    def statistic_voc(self):
        with open(self.content_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
            for line in self.lines:
                words = line.split(' ')
                if self.max_sentence_length < len(words):
                    self.max_sentence_length = len(words)

    def load(self):
        for i, line in enumerate(self.lines):
            ls = line.split('\t')
            # for sentence in ls:
            words = ls[0].split(" ")
            self.x_array_length[i] = len(words)
            for j, word in enumerate(words):
                self.x_array[i, j] = self.word_to_id[word.strip()]
            words = ls[1].split(" ")
            self.y_array_length[i] = len(words)
            for j, word in enumerate(words):
                self.y_array[i, j] = self.word_to_id[word.strip()]

    def train_data(self, batch_nub):
        if self.iternub > self.max_iter:
            return None
        for _ in range(batch_nub):
            if self.train_batch_index > self.train_test_index:
                self.train_batch_index = 0
                self.iternub += 1
            yield {
                    'x_data': self.x_array[self.train_batch_index: (self.train_batch_index + self.batch_size)],
                    'y_data': self.y_array[self.train_batch_index: (self.train_batch_index + self.batch_size)],
                    'x_data_length': self.x_array_length[self.train_batch_index: (self.train_batch_index + self.batch_size)],
                    'y_data_length': self.y_array_length[self.train_batch_index: (self.train_batch_index + self.batch_size)],
                   }
            self.train_batch_index += self.batch_size

    def test_data(self):
        yield{
                'x_data': self.x_array[self.train_test_index:],
                'y_data': self.y_array[self.train_test_index:],
                'x_data_length': self.x_array_length[self.train_test_index:],
                'y_data_length': self.y_array_length[self.train_test_index:],
        }

# todo x_data 应该增加一个维度表示对话 最长考虑5句对话？？？？填充？？？还是一个list？？？？
