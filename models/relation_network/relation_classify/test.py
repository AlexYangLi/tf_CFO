
# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: inference.py

@time: 2018/4/12 15:04

@desc:

"""

import os
import sys
import pickle
from gen_relclassify_data import read_data
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def compute_accuracy(model_path, data):
    """"compute top k hits accuracy"""

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    graph = tf.get_default_graph()

    # input
    question_ids = graph.get_tensor_by_name('question_ids:0')
    sequence_len = graph.get_tensor_by_name('sequence_len:0')
    pos_rel_ids = graph.get_tensor_by_name('pos_rel_ids:0')
    gru_keep_prob = graph.get_tensor_by_name('gru_keep_prob:0')

    # output
    accuracy = graph.get_tensor_by_name('accuracy:0')

    q_word_ids, q_seq_len, q_pos_rel_ids = data
    feed_dict = {question_ids: q_word_ids,
                 sequence_len: q_seq_len,
                 pos_rel_ids: q_pos_rel_ids,
                 gru_keep_prob: 1.0}
    acc =  sess.run(accuracy, feed_dict=feed_dict)

    print('accuracy: %f' % acc)


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'

    word2idx = pickle_load('../../../data/fb_word2idx.pkl')
    relation2idx = pickle_load('../../../data/FB5M_relation2idx.pkl')
    test_data = read_data(sys.argv[2], word2idx, relation2idx)

    compute_accuracy(sys.argv[1], test_data)


