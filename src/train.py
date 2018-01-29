# -*- coding: utf-8 -*-


import tensorflow as tf
import word2vec
import numpy as np

vec_path = r'F:\python_workspace\dialog_wechat\corpus\dialog_datas\sentence_vocbulart.txt.phrases.bin'
w2v = word2vec.load(vec_path)
voc = w2v.vocab
vec = w2v.vectors

for i, j in enumerate(voc):
    print(i, j)
# initializer = word2vec.load(vec_path).vectors.astype(np.float32)

# print(initializer[0])










