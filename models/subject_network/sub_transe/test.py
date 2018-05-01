# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: inference.py

@time: 2018/4/18 8:53

@desc:

"""

import os
import sys
import pickle
import numpy as np
from gen_data import read_data
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    pos_sub_ids = graph.get_tensor_by_name('pos_sub_ids:0')
    gru_keep_prob = graph.get_tensor_by_name('gru_keep_prob:0')

    # output
    pos_similarity = graph.get_tensor_by_name('pos_similarity:0')

    q_word_ids, q_seq_len, q_pos_sub_ids, cand_pos_ids = data
    top1 = top3 = top5 = top10 = 0
    for i in range(len(q_word_ids)):
        if i % 1000 == 0:
            print(i, top1)
        score = {}

        # compute score for each candidate relation
        q_word_ids_npy = np.zeros([1, 60])
        q_word_ids_npy[0, :len(q_word_ids[i])] = q_word_ids[i]
        mul_q_word_ids_npy = np.tile(q_word_ids_npy, (len(cand_pos_ids[i]), 1))

        mul_q_seq_len = np.tile(q_seq_len[i:i + 1], len(cand_pos_ids[i]))
        # print(mul_q_word_ids_npy.shape, mul_q_seq_len.shape, len(cand_pos_ids[i]))
        feed_dict = {question_ids: mul_q_word_ids_npy,
                     sequence_len: mul_q_seq_len,
                     pos_sub_ids: cand_pos_ids[i],
                     gru_keep_prob: 1.0
                     }
        similarity = sess.run(pos_similarity, feed_dict=feed_dict)

        for j in range(len(cand_pos_ids[i])):
            score[cand_pos_ids[i][j]] = similarity[j]
        # rank
        sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
        sorted_rel = [score[0] for score in sorted_score]
        if q_pos_sub_ids[i] in sorted_rel[:1]:
            top1 += 1
        if q_pos_sub_ids[i] in sorted_rel[:3]:
            top3 += 1
        if q_pos_sub_ids[i] in sorted_rel[:5]:
            top5 += 1
        if q_pos_sub_ids[i] in sorted_rel[:10]:
            top10 += 1
    print('accuracy: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' % (top1 / len(q_word_ids), top3 / len(q_word_ids),
                                                                      top5 / len(q_word_ids), top10 / len(q_word_ids)))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'

    word2idx = pickle_load('../../../data/fb_word2idx.pkl')
    subject2idx = pickle_load('../../../data/FB5M_subject2idx.pkl')
    idx2subject = pickle_load('../../../data/FB5M_idx2subject.pkl')
    _, test_data_metric = read_data(sys.argv[2], word2idx, subject2idx)

    compute_accuracy(sys.argv[1], test_data_metric)


