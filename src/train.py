# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from src import data_load
from src.model import ChatModel


model_dir = r'..\model'
content_path = r'..\corpus\dialog_datas\mini.sentence_dialog.txt'
content_path = r'..\corpus\dialog_datas\sentence_dialog.txt'
voc_path = r'..\corpus\dialog_datas\voc'
num_word = 26102
embedding_dim = 64
max_epoch = 1000

encoder_rnn_state_size = 100
decoder_rnn_state_size = 100
attention_num_units = 100
attention_depth = 100
beam_width = 5
learning_rate = 0.01
decay_steps = 1e4
decay_factor = 0.3
minimum_learning_rate = 1e-5


batch_data = data_load.DataLoader(content_path=content_path, voc_path=voc_path)
batch_size = 128


def run_train():
    max_iteration = batch_data.max_sentence_length + 1
    chat_model = ChatModel(batch_size=batch_size, max_iteration=max_iteration,
                           embedding_dim=embedding_dim)
    decoder_results, seq_loss = chat_model.encoder_decoder_graph(input_batch=None)

    loss_summary = tf.summary.scalar("loss", seq_loss)
    train_summary_op = tf.summary.merge([loss_summary])

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

        writer = tf.summary.FileWriter(r"../logs/", sess.graph)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))

        for epoch in range(max_epoch):
            train_data = batch_data.train_data(batch_size)
            for batch, data_dict in enumerate(train_data):
                feed_dict = chat_model.make_feed_dict(data_dict)
                # _decoder_outputs = sess.run(decoder_results['decoder_outputs'], feed_dict)
                _, decoder_result_ids_, loss_value_, train_summary_op_ = \
                    sess.run([train_op, decoder_results['decoder_result_ids'], seq_loss, train_summary_op], feed_dict)
                if epoch % 4 == 0:
                    writer.add_summary(train_summary_op_, epoch)
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss_value_))
            if epoch % 4 == 0:
                saver.save(sess, os.path.join(model_dir, 'chat'), global_step=epoch)


def main(_):
    run_train()

if __name__ == '__main__':
    tf.app.run()
