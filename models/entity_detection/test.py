# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: inference.py

@time: 2018/4/6 10:21

@desc:

"""

import sys
import os
import pickle
import tensorflow as tf
from gen_seqlabeling_data import read_data_for_test

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def gen_test_result(model_type, model_path, test_data):
    q_lineid, q_word_ids, q_seq_len, q_labels = test_data

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        graph = tf.get_default_graph()

        # input
        question_word_ids = graph.get_tensor_by_name('question_word_ids:0')
        sequence_len = graph.get_tensor_by_name('sequence_len:0')
        rnn_keep_prob = graph.get_tensor_by_name('rnn_keep_prob:0')
        labels = graph.get_tensor_by_name('label:0')

        # output
        if model_type == 'lstm' or model_type == 'bilstm':
            predict = graph.get_tensor_by_name('predict:0')
        else:
            predict = graph.get_tensor_by_name('ReverseSequence_1:0')    # just print(tensor) to get tensor's name
        macro_precision = graph.get_tensor_by_name('macro_precision:0')
        macro_recall = graph.get_tensor_by_name('macro_recall:0')
        macro_f1 = graph.get_tensor_by_name('macro_f1:0')

        feed_dict = {question_word_ids: q_word_ids, sequence_len:q_seq_len,
                     labels: q_labels, rnn_keep_prob: 1.0, }
        test_predict, test_macro_precision, test_macro_recall, test_macro_f1 = sess.run([predict, macro_precision,
                                                                                         macro_recall, macro_f1],
                                                                                        feed_dict=feed_dict)
        model_name = os.path.basename(model_path)
        print('%s test_precision: %f test_recall: %f, test_f1: %f' % (model_name, test_macro_precision,
                                                                      test_macro_recall, test_macro_f1))
        save_path = 'test_result/' + model_name + '_test_result.csv'
        with open(save_path, 'w') as writer:
            for i in range(len(q_lineid)):
                writer.write(q_lineid[i] + '\t')
                for j in range(int(q_seq_len[i])):
                    writer.write(str(test_predict[i][j]))
                    if j < int(q_seq_len[i]) - 1:
                        writer.write(' ')
                    else:
                        writer.write('\n')


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'argument error!'

    word2idx = pickle_load('../../data/fb_word2idx.pkl')
    test_data = read_data_for_test('../../data/test.csv', word2idx)

    gen_test_result(sys.argv[1], sys.argv[2], test_data)


