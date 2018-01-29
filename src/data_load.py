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
        word2id[i] = j
        id2word[j] = i
    return word2id, id2word, vectors


class DataLoader(object):

    iternub = 1
    max_sentence_length = 0

    def __init__(self, batch_size=30
                 , content_path=r'./data/content_cut'
                 , vec_bin_path=r'..\corpus\dialog_datas\sentence_vocbulart.txt.phrases.bin'
                 , max_iter=50):

        self.word_to_id, self.id_to_word, self.vectors = load_w2vec(vec_bin_path)
        self.datas = []
        self.lines = []

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.content_path = content_path
        self.statistic_voc()
        self.text_array = np.zeros([len(self.lines), self.max_sentence_length], dtype=np.int32)
        self.train_test_index = int(len(self.lines)/2)
        self.train_batch_index = 0
        self.test_batch_index = self.train_test_index
        self.load()

    def statistic_voc(self):
        with open(self.content_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
            for line in self.lines:
                line = line.split('\t')[1]
                words = [word.strip() for word in line.split(' ')]
                if self.max_sentence_length < len(words):
                    self.max_sentence_length = len(words)

    def load(self):
        for i, line in enumerate(self.lines):
            line = line.split('\t')[1]
            words = line.split(" ")
            for j, word in enumerate(words):
                self.text_array[i, j] = self.word_to_id[word.strip()]

    def train_batchs(self, batch_nub):
        if self.iternub > self.max_iter:
            return None
        for _ in range(batch_nub):
            if self.train_batch_index > self.train_test_index:
                self.train_batch_index = 0
                self.iternub += 1
            yield self.text_array[self.train_batch_index: (self.train_batch_index + self.batch_size)]
            self.train_batch_index += self.batch_size

    def test_batchs(self, batch_nub):
        if self.iternub > self.max_iter:
            return None
        for _ in range(batch_nub):
            if self.test_batch_index > len(self.lines):
                self.test_batch_index = self.train_test_index
                self.iternub += 1
            yield self.text_array[self.test_batch_index: (self.test_batch_index + self.batch_size)]
            self.test_batch_index += self.batch_size