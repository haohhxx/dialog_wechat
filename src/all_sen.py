# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
    BasicDecoder, BeamSearchDecoder, dynamic_decode, \
    TrainingHelper, sequence_loss, tile_batch, \
    BahdanauAttention, LuongAttention


class Utterance2Seq(object):
    batch_size = 30
    encoder_inputs = tf.placeholder(shape=(batch_size, 5, None), dtype=tf.int32, name='encoder_inputs')
    encoder_lengths = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='encoder_lengths')
    # batch_size, max_time
    decoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')
    decoder_lengths = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='decoder_inputs')


