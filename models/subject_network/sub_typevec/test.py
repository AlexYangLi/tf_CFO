# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: inference.py

@time: 2018/4/18 9:03

@desc:

"""

import os
import sys
import pickle
import numpy as np
from gen_data import read_data
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def get_typevec(type_ids):
        type_vec = np.zeros((len(type_ids), 500))
        for i in range(len(type_ids)):
            for type_id in type_ids[i]:
                type_vec[i][type_id] = 1
        return type_vec


def compute_accuracy(model_path, data, score_func):
    """"compute top k hits accuracy"""

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    graph = tf.get_default_graph()

    # input
    question_ids = graph.get_tensor_by_name('question_ids:0')
    sequence_len = graph.get_tensor_by_name('sequence_len:0')
    sub_typevec = graph.get_tensor_by_name('sub_typevec:0')
    gru_keep_prob = graph.get_tensor_by_name('gru_kepp_prob:0')

    # output
    if score_func == 'sum':
        score = graph.get_tensor_by_name('sum_score:0')
    else:
        score = graph.get_tensor_by_name('mean_score:0')

    q_word_ids, q_seq_len, q_sub_typeids, q_cand_sub_ids, q_can_sub_typeids = data
    top1 = top3 = top5 = top10 = 0
    data_size = len(q_word_ids)
    for i in range(data_size):
        q_score = {}

        num_cand = len(q_cand_sub_ids[i])
        pos_sub_idx = q_cand_sub_ids[i][0]  # index 0 will always be positive

        # compute score for each candidate subject
        mul_q_word_ids = np.tile(q_word_ids[i], (num_cand, 1))
        mul_q_seq_len = np.tile(q_seq_len[i], num_cand)
        feed_dict = {question_ids: mul_q_word_ids,
                     sequence_len: mul_q_seq_len,
                     sub_typevec: get_typevec(q_can_sub_typeids[i]),
                     gru_keep_prob: 1.0}
        result = sess.run(score, feed_dict=feed_dict)

        for j in range(num_cand):
            q_score[q_cand_sub_ids[i][j]] = result[j]

        # rank by score
        sorted_q_score = sorted(q_score.items(), key=lambda x: x[1], reverse=True)
        sorted_sub = [s[0] for s in sorted_q_score]

        if pos_sub_idx in sorted_sub[:1]:
            top1 += 1
        if pos_sub_idx in sorted_sub[:3]:
            top3 += 1
        if pos_sub_idx in sorted_sub[:5]:
            top5 += 1
        if pos_sub_idx in sorted_sub[:10]:
            top10 += 1

    print('%s score rank: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' %
          (score_func, top1 / data_size, top3 / data_size, top5 / data_size, top10 / data_size))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'
    
    
    word2idx = pickle_load('../../../../data/fb_word2idx.pkl')
    idx2subject = pickle_load('../../../../data/FB5M_idx2subject.pkl')
    subject2idx = pickle_load('../../../../data/FB5M_subject2idx.pkl')
    subject2type = pickle_load('../../../../data/trim_subject2type.pkl')
    type2idx = pickle_load('../../../../data/FB5M_type2idx.pkl')

    test_data = read_data(sys.argv[2], word2idx, idx2subject, subject2idx, subject2type, type2idx)

    compute_accuracy(sys.argv[1], test_data, 'sum')
    compute_accuracy(sys.argv[1], test_data, 'mean')

