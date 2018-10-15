"""
2018-7-16
增加适用于pyTorch 的 char-embedding Vocab
"""
# -*- coding:utf8 -*-

import numpy as np
import torch
import torch.nn as nn


class Vocab(object):

    def __init__(self, filename=None, initial_terms=None, lower=False):
        self.id2term = {}
        self.term2id = {}
        self.term_frequent = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_term = '<pad>'
        self.unk_term = '<unk>'
        self.eos_term = '<eos>'
        self.sos_term = '<sos>'

        self.initial_terms = initial_terms if initial_terms is not None else []
        self.initial_terms.extend([self.pad_term, self.unk_term, self.eos_term, self.sos_term])
        for term in self.initial_terms:
            self.add(term)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2term)

    def load_from_file(self, file_path):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            term = line.rstrip('\n')
            self.add(term)

    def get_id(self, term):
        """
        gets the id of a term, returns the id of unk term if term is not in vocab
        Args:
            term: a string indicating the word
        Returns:
            an integer
        """
        term = term.lower() if self.lower else term
        try:
            return self.term2id[term]
        except KeyError:
            return self.term2id[self.unk_term]

    def get_term(self, idx):
        """
        gets the term corresponding to idx, returns unk term if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a term string
        """
        try:
            return self.id2term[idx]
        except KeyError:
            return self.unk_term

    def add(self, term, count=1):
        """
        adds the term to vocab
        Args:
            term: a string
            count: a num indicating the count of the term to add, default is 1
        Returns:
            id of term
        """
        term = term.lower() if self.lower else term
        if term in self.term2id:
            idx = self.term2id[term]
        else:
            idx = len(self.id2term)
            self.id2term[idx] = term
            self.term2id[term] = idx
        if count > 0:
            if term in self.term_frequent:
                self.term_frequent[term] += count
            else:
                self.term_frequent[term] = count
        return idx

    def filter_terms_by_cnt(self, min_count):
        """
        filter the terms in vocab by their count
        Args:
            min_count: terms with frequency less than min_cnt is filtered
        """
        filtered_terms = [term for term in self.term2id if self.term_frequent[term] >= min_count]
        # rebuild the term x id map
        self.term2id = {}
        self.id2term = {}
        for term in self.initial_terms:
            self.add(term, count=0)
        for term in filtered_terms:
            self.add(term, count=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each term
        Args:
            embed_dim: the size of the embedding for each term
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for term in [self.pad_term, self.unk_term, self.eos_term]:
            self.embeddings[self.get_id(term)] = np.zeros([self.embed_dim])

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        terms not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                contents = line.strip().split(" ")
                term = contents[0]
                if term not in self.term2id:
                    continue
                trained_embeddings[term] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_terms = trained_embeddings.keys()
        # rebuild the term x id map
        self.term2id = {}
        self.id2term = {}
        for term in self.initial_terms:
            self.add(term, count=0)
        for term in filtered_terms:
            self.add(term, count=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for term in self.term2id.keys():
            if term in trained_embeddings:
                self.embeddings[self.get_id(term)] = trained_embeddings[term]

    def convert_to_ids(self, terms):
        """
        Convert a list of terms to ids, use unk_term if the term is not in vocab.
        Args:
            terms: a list of term
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in terms]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to terms, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of terms
        """
        terms = []
        for i in ids:
            terms += [self.get_term(i)]
            if stop_id is not None and i == stop_id:
                break
        return terms


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim

        self.lin1 = nn.Linear(h_dim, h_dim, bias=True)  # 24
        self.relu_layer = nn.ReLU(True)
        self.lin_out = nn.Linear(h_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

        # nn.init.uniform_(self.lin1.weight, a=-0.1, b=0.1)
        # nn.init.uniform_(self.lin_out.weight, a=-0.1, b=0.1)
        # nn.init.uniform_(self.lin1.bias, a=-0.1, b=0.1)
        # nn.init.uniform_(self.lin_out.bias, a=-0.1, b=0.1)

    def forward(self, encoder_outputs):
        # b_size = encoder_outputs.size(0)
        # encoder_outputs = encoder_outputs.view(-1, self.h_dim)
        l1out = self.lin1(encoder_outputs)
        l1out = self.relu_layer(l1out)
        l2out = self.lin_out(l1out)
        attn_ene = self.softmax(l2out)
        # attn_ene = self.main(encoder_outputs)  # (b, s, h) -> (b * s, 1)
        # attn_ene = attn_ene.view(b_size, -1)
        return attn_ene


class CharEmbedding(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = int(in_size/2)
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=True,
                          num_layers=self.num_layers,
                          hidden_size=self.hidden_size, batch_first=False,
                          dropout=0.3)
        self.h = torch.randn(self.num_layers * 2, 1, self.hidden_size)
        # self.out_size = self.hidden_size * self.num_layers * 2

    def forward(self, input):
        # (l, b, in_size) = input.size()
        # input = nn.Dropout(0.3)(input)
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h


class CharVocab(Vocab):

    def __init__(self, filename=None, initial_terms=None, lower=False):
        super().__init__(filename, initial_terms, lower)
        self.id2char = {}
        self.char2id = {}
        self.char_embeddings = None
        self.word_char_embeddings = None

    def add(self, term, count=1):
        """
        adds the term to vocab
        Args:
            term: a string
            count: a num indicating the count of the term to add, default is 1
        Returns:
            id of term
        """
        term = term.lower() if self.lower else term
        if term in self.term2id:
            idx = self.term2id[term]
        else:
            idx = len(self.id2term)
            self.id2term[idx] = term
            self.term2id[term] = idx
        if count > 0:
            if term in self.term_frequent:
                self.term_frequent[term] += count
            else:
                self.term_frequent[term] = count

        if term not in self.initial_terms:
            for char in term:
                if char not in self.char2id.keys():
                    idc = len(self.id2char)
                    self.id2char[idc] = char
                    self.char2id[char] = idc
        return idx

    def load_char_embeddings(self, embedding_dim):
        """
        loads the pretrained embeddings from embedding_path,
        terms not in pretrained embeddings will be filtered
        """
        self.char_embeddings = np.random.rand(self.char_size(), embedding_dim)
        self.word_char_embeddings = np.random.rand(self.size(), embedding_dim)
        for term in [self.pad_term, self.unk_term, self.eos_term]:
            self.word_char_embeddings[self.get_id(term)] = np.zeros([self.embed_dim])
        # self.word_char_embeddings = self.embeddings
        char_emb = CharEmbedding(embedding_dim)
        for term in self.term2id.keys():
            if term not in self.initial_terms:
                char_ids = [self.char_embeddings[self.char2id[char]] for char in term]
                chars = torch.FloatTensor(char_ids)
                chars = torch.unsqueeze(chars, 1)

                hidden = char_emb(chars)
                self.word_char_embeddings[self.get_id(term)] = hidden.data.numpy()
                # self.embeddings[self.get_id(term)] = np.zeros([self.embed_dim])
            # self.embeddings = np.random.rand(self.size(), embedding_dim)

    def char_size(self):
        """
        get the size of char vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2char)







