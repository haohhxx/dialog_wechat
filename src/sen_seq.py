# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
    BasicDecoder, BeamSearchDecoder, dynamic_decode, \
    TrainingHelper, sequence_loss, tile_batch, \
    BahdanauAttention, LuongAttention


class Sen2Seq(object):

    num_word = 0
    embedding_dim = 128
    batch_size = 30
    max_epoch = 100
    max_iteration = 10
    encoder_rnn_state_size = 100
    decoder_rnn_state_size = 100
    attention_num_units = 100
    attention_depth = 100
    beam_width = 5
    learning_rate = 0.01
    decay_steps = 1e4
    decay_factor = 0.3
    minimum_learning_rate = 1e-5
    PAD = 0
    EOS = 1

    def __init__(self):
        pass

    def _build_encoder(self, encoder_inputs, word_embedding, encoder_lengths):
        with tf.variable_scope('word_embedding'):
            # word_embedding = tf.get_variable(
            #     name="word_embedding",
            #     shape=(num_word, embedding_dim),
            #     initializer=xavier_initializer(),
            #     dtype=tf.float32
            # )
            # batch_size, max_time, embed_dims
            encoder_input_vectors = tf.nn.embedding_lookup(word_embedding, encoder_inputs)
        with tf.variable_scope('encoder'):
            encoder_cell = LSTMCell(self.encoder_rnn_state_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell,
                encoder_input_vectors,
                sequence_length=encoder_lengths,
                time_major=False,
                dtype=tf.float32
            )
            encoder_final_state.set_shape([self.batch_size, self.encoder_rnn_state_size])

    def _word_level_attention(self, encoder_outputs, word_embedding, encoder_lengths):
        with tf.variable_scope('attention'):
            attention_mechanism = BahdanauAttention(
                self.attention_num_units,
                encoder_outputs,
                encoder_lengths,
                name="attention_fn"
            )
            beam_attention_mechanism = BahdanauAttention(
                attention_num_units,
                tiled_encoder_outputs,
                tiled_encoder_lengths,
                name="attention_fn"
            )






