# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import jieba

from src import data_load
from src.model.model_atten import ChatModel as ChatModelBirnnAtten
from src.model.model_birnn import ChatModel as ChatModelBirnn

model_dir = r'..\model'
content_path = r'..\corpus\dialog_datas\sentence_dialog.txt'
voc_path = r'..\corpus\dialog_datas\voc'

embedding_dim = 64

batch_data = data_load.DataLoader(content_path=content_path, voc_path=voc_path)
word_to_id = batch_data.word_to_id
id_to_word = batch_data.id_to_word

num_word = len(id_to_word)
batch_size = 1
max_iteration = batch_data.max_sentence_length + 1


def load_model():
    chat_model = ChatModelBirnn(batch_size=batch_size, max_iteration=max_iteration, num_word=num_word,
                                embedding_dim=embedding_dim)
    decoder_results, seq_loss = chat_model.encoder_decoder_graph(input_batch=None)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)
        while 1:
            x_str = str(input("in x data:\n"))
            if x_str == "exit":
                break
            feed_dict = build_dict(x_str=x_str, chat_model=chat_model)
            decoder_outputs_, beam_decoder_result_ids_, beam_decoder_sequence_outputs_,  = \
                sess.run([
                          decoder_results['decoder_outputs'],
                          decoder_results['beam_decoder_result_ids'],
                          decoder_results['beam_decoder_sequence_outputs'],

                          ],
                         feed_dict)
            out = [id_to_word[wids[0]] for wids in beam_decoder_result_ids_[0]]
            print(out)
            out = [id_to_word[wids[1]] for wids in beam_decoder_result_ids_[0]]
            print(out)
            out = [id_to_word[wids[2]] for wids in beam_decoder_result_ids_[0]]
            print(out)
            out = [id_to_word[wids[3]] for wids in beam_decoder_result_ids_[0]]
            print(out)
            out = [id_to_word[wids[4]] for wids in beam_decoder_result_ids_[0]]
            print(out)

            # print(decoder_outputs_)
            # print(out)
            print(beam_decoder_result_ids_)
            print(beam_decoder_sequence_outputs_)


def build_dict(x_str, chat_model):
    input_str = jieba.cut(str(x_str))
    x_array = np.zeros([1, max_iteration], dtype=np.int32)
    for i, word in enumerate(input_str):
        x_array[0, i] = word_to_id[word]
    x_array_length = np.array([1], dtype=np.int32)
    data_dict = {
        'encoder_inputs': x_array,
        'decoder_inputs': x_array,
        'encoder_lengths': x_array_length,
        'decoder_lengths': x_array_length,
    }
    feed_dict = chat_model.make_feed_dict(data_dict)
    return feed_dict


def main(_):
    load_model()


if __name__ == '__main__':
    tf.app.run()
