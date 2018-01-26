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

    def _build_utterance_attention(self, sen_attention):
        with tf.variable_scope('utterance_attention'):
            attention_mechanism = BahdanauAttention(
                self.attention_num_units,
                encoder_outputs,
                encoder_lengths,
                name="attention_fn"
            )
            
            decoder_cell = AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                attention_layer_size=self.attention_depth,
                output_attention=True,
            )
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)\
                                                .clone(cell_state=encoder_final_state)
        return decoder_initial_state