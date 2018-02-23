# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
    BasicDecoder, BeamSearchDecoder, dynamic_decode, \
    TrainingHelper, sequence_loss, tile_batch, \
    BahdanauAttention, LuongAttention


class ChatModel(object):
    def __init__(self, batch_size, max_iteration, num_word , embedding_dim=128):
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.num_word = num_word
        self._inputs = {}
        self.embedding_dim = embedding_dim

    PAD = 0 # 填充标记
    EOS = 1 # 结束标记
    encoder_rnn_state_size = 100
    decoder_rnn_state_size = 100
    attention_num_units = 100
    attention_depth = 100
    beam_width = 5
    minimum_learning_rate = 1e-5

    def _build_inputs(self, input_batch):
        if input_batch is None:
            self._inputs['encoder_inputs'] = tf.placeholder(
                shape=(self.batch_size, None),  # batch_size, max_time
                dtype=tf.int32,
                name='encoder_inputs'
            )
            self._inputs['encoder_lengths'] = tf.placeholder(
                shape=(self.batch_size,),
                dtype=tf.int32,
                name='encoder_lengths'
            )
            self._inputs['decoder_inputs'] = tf.placeholder(
                shape=(self.batch_size, None),  # batch_size, max_time
                dtype=tf.int32,
                name='decoder_inputs'
            )
            self._inputs['decoder_lengths'] = tf.placeholder(
                shape=(self.batch_size,),
                dtype=tf.int32,
                name='decoder_lengths'
            )

        else:
            encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = input_batch
            encoder_inputs.set_shape([self.batch_size, None])
            decoder_inputs.set_shape([self.batch_size, None])
            encoder_lengths.set_shape([self.batch_size])
            decoder_lengths.set_shape([self.batch_size])

            self._inputs = {
                'encoder_inputs': encoder_inputs,
                'encoder_lengths': encoder_lengths,
                'decoder_inputs': decoder_inputs,
                'decoder_lengths': decoder_lengths
            }

        return self._inputs['encoder_inputs'], self._inputs['encoder_lengths'], self._inputs['decoder_inputs'], self._inputs['decoder_lengths']

    def make_feed_dict(self, data_dict):
        feed_dict = {}
        for key in data_dict.keys():
            try:
                feed_dict[self._inputs[key]] = data_dict[key]
            except KeyError:
                raise ValueError('Unexpected argument in input dictionary!')
        return feed_dict

    def encoder_decoder_graph(self, input_batch):
        encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = self._build_inputs(input_batch)
        # self.encoder_inputs = encoder_inputs

        with tf.variable_scope('word_embedding'):
            word_embedding = tf.get_variable(
                name="word_embedding",
                shape=(self.num_word, self.embedding_dim),
                initializer=xavier_initializer(),
                dtype=tf.float32
            )
            # batch_size, max_time, embed_dims
            encoder_input_vectors = tf.nn.embedding_lookup(word_embedding, encoder_inputs)

        with tf.variable_scope('encoder'):
            encoder_cell = LSTMCell(self.encoder_rnn_state_size)
            (fw_output, bw_output), (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell, encoder_cell,
                encoder_input_vectors,
                sequence_length=encoder_lengths,
                time_major=False,
                dtype=tf.float32
            )
            encoder_outputs = tf.concat([fw_output, bw_output], 2)
            # if isinstance(fw_final_state, LSTMStateTuple):
            encoder_state_c = tf.concat([fw_final_state.c, bw_final_state.c], 1)
            encoder_state_h = tf.concat([fw_final_state.h, bw_final_state.h], 1)
            encoder_state_c.set_shape([self.batch_size, self.encoder_rnn_state_size * 2])
            encoder_state_h.set_shape([self.batch_size, self.encoder_rnn_state_size * 2])

            encoder_final_state = LSTMStateTuple(encoder_state_c, encoder_state_h)

        tiled_batch_size = self.batch_size * self.beam_width
        with tf.variable_scope('decoder_cell'):
            self.decoder_rnn_state_size *= 2
            decoder_cell = LSTMCell(self.decoder_rnn_state_size)
            original_decoder_cell = decoder_cell

            with tf.variable_scope('beam_inputs'):
                tiled_encoder_outputs = tile_batch(encoder_outputs, self.beam_width)
                tiled_encoder_lengths = tile_batch(encoder_lengths, self.beam_width)

                tiled_encoder_final_state_c = tile_batch(encoder_final_state.c, self.beam_width)
                tiled_encoder_final_state_h = tile_batch(encoder_final_state.h, self.beam_width)
                tiled_encoder_final_state = LSTMStateTuple(tiled_encoder_final_state_c, tiled_encoder_final_state_h)

            with tf.variable_scope('attention'):
                attention_mechanism = BahdanauAttention(
                    self.attention_num_units,
                    encoder_outputs,
                    encoder_lengths,
                    name="attention_fn"
                )
                decoder_cell = AttentionWrapper(
                    decoder_cell,
                    attention_mechanism,
                    attention_layer_size=self.attention_depth,
                    output_attention=True,
                )
                decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32) \
                    .clone(cell_state=encoder_final_state)

            with tf.variable_scope('attention', reuse=True):
                beam_attention_mechanism = BahdanauAttention(
                    self.attention_num_units,
                    tiled_encoder_outputs,
                    tiled_encoder_lengths,
                    name="attention_fn"
                )
                beam_decoder_cell = AttentionWrapper(
                    original_decoder_cell,
                    beam_attention_mechanism,
                    attention_layer_size=self.attention_depth,
                    output_attention=True
                )
                # beam_decoder_cell 单独的original_decoder_cell
                tiled_decoder_initial_state = beam_decoder_cell \
                    .zero_state(tiled_batch_size, tf.float32) \
                    .clone(cell_state=tiled_encoder_final_state)

            # with tf.variable_scope('word_embedding', reuse=True):
            #     word_embedding = tf.get_variable(name="word_embedding")

            with tf.variable_scope('decoder'):
                out_func = layers_core.Dense(self.num_word, use_bias=False)
                eoses = tf.ones([self.batch_size, 1], dtype=tf.int32) * self.EOS
                eosed_decoder_inputs = tf.concat([eoses, decoder_inputs], 1)
                embed_decoder_inputs = tf.nn.embedding_lookup(word_embedding, eosed_decoder_inputs)

                training_helper = TrainingHelper(
                    embed_decoder_inputs,
                    decoder_lengths + 1
                )
                decoder = BasicDecoder(
                    decoder_cell,
                    training_helper,
                    decoder_initial_state,
                    output_layer=out_func,
                )
                decoder_outputs, decoder_state, decoder_sequence_lengths = dynamic_decode(
                    decoder,
                    scope=tf.get_variable_scope(),
                    maximum_iterations=self.max_iteration
                )

                tf.get_variable_scope().reuse_variables()
                start_tokens = tf.ones([self.batch_size], dtype=tf.int32) * self.EOS
                beam_decoder = BeamSearchDecoder(
                    beam_decoder_cell,
                    word_embedding,
                    start_tokens,
                    self.EOS,
                    tiled_decoder_initial_state,
                    self.beam_width,
                    output_layer=out_func,
                )
                beam_decoder_outputs, beam_decoder_state, beam_decoder_sequence_lengths = dynamic_decode(
                    beam_decoder,
                    scope=tf.get_variable_scope(),
                    maximum_iterations=self.max_iteration
                )

            decoder_results = {
                'decoder_outputs': decoder_outputs[0],
                'decoder_result_ids': decoder_outputs[1],
                'decoder_state': decoder_state,
                'decoder_sequence_lengths': decoder_sequence_lengths,
                'beam_decoder_result_ids': beam_decoder_outputs.predicted_ids,
                'beam_decoder_scores': beam_decoder_outputs.beam_search_decoder_output.scores,
                'beam_decoder_state': beam_decoder_state,
                'beam_decoder_sequence_outputs': beam_decoder_sequence_lengths
            }
            with tf.variable_scope('loss_target'):
                # build decoder output, with appropriate padding and mask
                # batch_size = batch_size
                pads = tf.ones([self.batch_size, 1], dtype=tf.int32) * self.PAD
                paded_decoder_inputs = tf.concat([decoder_inputs, pads], 1)
                max_decoder_time = tf.reduce_max(decoder_lengths) + 1
                decoder_target = paded_decoder_inputs[:, :max_decoder_time]

                decoder_eos = tf.one_hot(decoder_lengths, depth=max_decoder_time,
                                         on_value=self.EOS, off_value=self.PAD,
                                         dtype=tf.int32)
                decoder_target += decoder_eos
                decoder_loss_mask = tf.sequence_mask(decoder_lengths + 1, maxlen=max_decoder_time, dtype=tf.float32)

            with tf.variable_scope('loss'):
                decoder_logits = decoder_results['decoder_outputs']
                seq_loss = sequence_loss(
                    decoder_logits,
                    decoder_target,
                    decoder_loss_mask,
                    name='sequence_loss'
                )
            # self.decoder_results = decoder_results
            return decoder_results, seq_loss
