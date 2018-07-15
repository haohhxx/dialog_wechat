# -*- coding:utf8 -*-

import numpy as np


class Vocab(object):

    def __init__(self, filename=None, initial_terms=None, lower=False):
        self.id2term = {}
        self.term2id = {}
        self.term_frequent = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_term = '<blank>'
        self.unk_term = '<unk>'
        self.eos_term = '<eos>'

        self.initial_terms = initial_terms if initial_terms is not None else []
        self.initial_terms.extend([self.pad_term, self.unk_term, self.eos_term])
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
