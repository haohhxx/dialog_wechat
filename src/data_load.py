#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np


class DataLoader(object):
    def __init__(self, batch_size=30, content=r'./data/content_cut', max_iter=50):

        self.word_to_id = {}
        self.id_to_word = {}
        self.voc_size = {}
        self.datas = []
        self.words_as_set = []
        self.lines = []

        self.iternub = 1
        self.max_sentence_length = 0
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.content = content
        self.statistic_voc()
        self.text_array = np.zeros([len(self.lines), self.max_sentence_length], dtype=np.int32)
        self.train_test_index = int(len(self.lines)/2)
        self.train_batch_index = 0
        self.test_batch_index = self.train_test_index
        self.load()

    def statistic_voc(self):
        with open(self.content, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        for line in self.lines:
            line = line.split('\t')[1]
            words = [word.strip() for word in line.split(' ')]
            if self.max_sentence_length < len(words):
                self.max_sentence_length = len(words)
            self.words_as_set.extend(words)
        self.words_as_set = set(self.words_as_set)
        self.word_to_id = {w: i for i, w in enumerate(self.words_as_set)}
        self.id_to_word = {i: w for i, w in enumerate(self.words_as_set)}
        self.voc_size = len(self.words_as_set)

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