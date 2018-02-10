# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
    BasicDecoder, BeamSearchDecoder, dynamic_decode, \
    TrainingHelper, sequence_loss, tile_batch, \
    BahdanauAttention, LuongAttention

from src import data_load

batch_size = 30
batch_data = data_load.DataLoader(batch_size)

model_dir = r'./model'

num_word = 26102
embedding_dim = 128
max_epoch = 1000
max_iteration = batch_data.max_sentence_length+1
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

encoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')
encoder_lengths = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='encoder_lengths')
# batch_size, max_time
decoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')
decoder_lengths = tf.placeholder(shape=(batch_size,),  dtype=tf.int32, name='decoder_lengths')

encoder_inputs.set_shape([batch_size, None])
decoder_inputs.set_shape([batch_size, None])
encoder_lengths.set_shape([batch_size])
decoder_lengths.set_shape([batch_size])

with tf.variable_scope('word_embedding'):
    word_embedding = tf.get_variable(
        name="word_embedding",
        shape=(num_word, embedding_dim),
        initializer=xavier_initializer(),
        dtype=tf.float32
    )
    # batch_size, max_time, embed_dims
    encoder_input_vectors = tf.nn.embedding_lookup(word_embedding, encoder_inputs)

with tf.variable_scope('encoder'):
    encoder_cell = LSTMCell(encoder_rnn_state_size)
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
    encoder_state_c.set_shape([batch_size, encoder_rnn_state_size * 2])
    encoder_state_h.set_shape([batch_size, encoder_rnn_state_size * 2])

    encoder_final_state = LSTMStateTuple(encoder_state_c, encoder_state_h)

tiled_batch_size = batch_size * beam_width
with tf.variable_scope('decoder_cell'):
    decoder_rnn_state_size *= 2
    decoder_cell = LSTMCell(decoder_rnn_state_size)
    original_decoder_cell = decoder_cell

    with tf.variable_scope('beam_inputs'):
        tiled_encoder_outputs = tile_batch(encoder_outputs, beam_width)
        tiled_encoder_lengths = tile_batch(encoder_lengths, beam_width)

        tiled_encoder_final_state_c = tile_batch(encoder_final_state.c, beam_width)
        tiled_encoder_final_state_h = tile_batch(encoder_final_state.h, beam_width)
        tiled_encoder_final_state = LSTMStateTuple(tiled_encoder_final_state_c, tiled_encoder_final_state_h)

    with tf.variable_scope('attention'):
        attention_mechanism = BahdanauAttention(
            attention_num_units,
            encoder_outputs,
            encoder_lengths,
            name="attention_fn"
        )
        decoder_cell = AttentionWrapper(
            decoder_cell,
            attention_mechanism,
            attention_layer_size=attention_depth,
            output_attention=True,
        )
        decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_final_state)

    with tf.variable_scope('attention', reuse=True):
        beam_attention_mechanism = BahdanauAttention(
            attention_num_units,
            tiled_encoder_outputs,
            tiled_encoder_lengths,
            name="attention_fn"
        )
        beam_decoder_cell = AttentionWrapper(
            original_decoder_cell,
            beam_attention_mechanism,
            attention_layer_size=attention_depth,
            output_attention=True
        )
        # beam_decoder_cell 单独的original_decoder_cell
        tiled_decoder_initial_state = beam_decoder_cell\
            .zero_state(tiled_batch_size, tf.float32)\
            .clone(cell_state=tiled_encoder_final_state)

    # with tf.variable_scope('word_embedding', reuse=True):
    #     word_embedding = tf.get_variable(name="word_embedding")

    with tf.variable_scope('decoder'):
        out_func = layers_core.Dense(num_word, use_bias=False)
        eoses = tf.ones([batch_size, 1], dtype=tf.int32) * EOS
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
                    maximum_iterations=max_iteration
                )

        tf.get_variable_scope().reuse_variables()
        start_tokens = tf.ones([batch_size], dtype=tf.int32) * EOS
        beam_decoder = BeamSearchDecoder(
            beam_decoder_cell,
            word_embedding,
            start_tokens,
            EOS,
            tiled_decoder_initial_state,
            beam_width,
            output_layer=out_func,
        )
        beam_decoder_outputs, beam_decoder_state, beam_decoder_sequence_lengths = dynamic_decode(
            beam_decoder,
            scope=tf.get_variable_scope(),
            maximum_iterations=max_iteration
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
    pads = tf.ones([batch_size, 1], dtype=tf.int32) * PAD
    paded_decoder_inputs = tf.concat([decoder_inputs, pads], 1)
    max_decoder_time = tf.reduce_max(decoder_lengths) + 1
    decoder_target = paded_decoder_inputs[:, :max_decoder_time]

    decoder_eos = tf.one_hot(decoder_lengths, depth=max_decoder_time,
                             on_value=EOS, off_value=PAD,
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

with tf.variable_scope('train'):
    train_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(
        learning_rate,
        train_step,
        decay_steps,
        decay_factor,
        staircase=True
    )
    lr = tf.clip_by_value(lr, minimum_learning_rate, learning_rate, name='lr_clip')
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_variables = tf.trainable_variables()
    grads_vars = opt.compute_gradients(seq_loss, train_variables)
    for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=train_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train_step')

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    for epoch in range(max_epoch):
        for batch, data_dict in enumerate(batch_data.train_data()):
            feed_dict = {
                encoder_inputs: data_dict['x_data'],
                encoder_lengths: data_dict['x_data_length'],
                decoder_inputs: data_dict['y_data'],
                decoder_lengths: data_dict['y_data_length'],
            }
            # _decoder_outputs = sess.run(decoder_results['decoder_outputs'], feed_dict)
            _, decoder_result_ids_, loss_value_ = \
                sess.run([train_op, decoder_results['decoder_result_ids'], seq_loss], feed_dict)
            if epoch % 10 == 0:
                print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss_value_))

            saver.save(sess, os.path.join(model_dir, 'chat'), global_step=epoch)



