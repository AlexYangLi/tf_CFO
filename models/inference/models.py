# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: models.py

@time: 2018/4/22 10:03

@desc: models for inference, containing entity detection, relation network & subject network

"""

import numpy as np
import tensorflow as tf


class EntDect(object):
    def __init__(self, model_type, model_path):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = tf.train.import_meta_graph(model_path+'.meta')
        self.saver.restore(self.sess, model_path)

        # input
        self.question_word_ids = self.graph.get_tensor_by_name('question_word_ids:0')
        self.sequence_len = self.graph.get_tensor_by_name('sequence_len:0')
        self.rnn_keep_prob = self.graph.get_tensor_by_name('rnn_keep_prob:0')

        # output
        if model_type == 'lstm' or model_type == 'bilstm':
            self.predict = self.graph.get_tensor_by_name('predict:0')
        else:
            self.predict = self.graph.get_tensor_by_name('ReverseSequence_1:0')  # just print(tensor) to get tensor's name

    def infer(self, test_data):
        questions, q_word_ids, q_seq_len = test_data

        feed_dict = {self.question_word_ids: q_word_ids,
                     self.sequence_len: q_seq_len,
                     self.rnn_keep_prob: 1.0}
        predict = self.sess.run(self.predict, feed_dict=feed_dict)

        predict_tokens = []
        for i in range(len(questions)):
            tokens = questions[i].split(' ')
            pred_indexes = list(np.nonzero(predict[i])[0])
            predict_token = ' '.join([tokens[j] for j in pred_indexes])
            predict_tokens.append(predict_token)

        return predict_tokens


class RelNet(object):
    def __init__(self, model_path):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(model_path + '.meta')
        self.saver.restore(self.sess, model_path)
        self.graph = tf.get_default_graph()

        # input
        self.question_ids = self.graph.get_tensor_by_name('question_ids:0')
        self.sequence_len = self.graph.get_tensor_by_name('sequence_len:0')
        self.pos_rel_ids = self.graph.get_tensor_by_name('pos_rel_ids:0')
        self.gru_keep_prob = self.graph.get_tensor_by_name('gru_keep_prob:0')

        # output
        self.pos_similarity = self.graph.get_tensor_by_name('pos_similarity:0')

    def infer(self, test_data):
        q_word_ids, q_seq_len, cand_rel_ids = test_data

        scores = []
        data_size = q_word_ids.shape[0]
        for i in range(data_size):
            score = {}  # key: relation_id, value: score

            # compute score for each candidate relation
            dup_q_word_ids = np.tile(q_word_ids[i], (len(cand_rel_ids[i]), 1))
            dup_q_seq_len = np.tile(q_seq_len[i], len(cand_rel_ids[i]))

            feed_dict = {self.question_ids: dup_q_word_ids,
                         self.sequence_len: dup_q_seq_len,
                         self.pos_rel_ids: cand_rel_ids[i],
                         self.gru_keep_prob: 1.0
                         }

            similarity = self.sess.run(self.pos_similarity, feed_dict=feed_dict)

            for j in range(len(cand_rel_ids[i])):
                score[cand_rel_ids[i][j]] = similarity[j]

            scores.append(score)

        return scores


class SubTransE(object):
    def __init__(self, model_path):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(model_path+'.meta')
        self.saver.restore(self.sess, model_path)
        self.graph = tf.get_default_graph()

        # input
        self.question_ids = self.graph.get_tensor_by_name('question_ids:0')
        self.sequence_len = self.graph.get_tensor_by_name('sequence_len:0')
        self.pos_sub_ids = self.graph.get_tensor_by_name('pos_sub_ids:0')
        self.gru_keep_prob = self.graph.get_tensor_by_name('gru_keep_prob:0')

        # output
        self.pos_similarity = self.graph.get_tensor_by_name('pos_similarity:0')

    def infer(self, test_data):
        q_word_ids, q_seq_len, cand_sub_ids = test_data

        scores = []
        data_size = q_word_ids.shape[0]
        for i in range(data_size):
            score = {}  # key: subject_id, value: score

            # compute score for each candidate subject
            dup_q_word_ids = np.tile(q_word_ids[i], (len(cand_sub_ids[i]), 1))
            dup_q_seq_len = np.tile(q_seq_len[i], len(cand_sub_ids[i]))

            feed_dict = {self.question_ids: dup_q_word_ids,
                         self.sequence_len: dup_q_seq_len,
                         self.pos_sub_ids: cand_sub_ids[i],
                         self.gru_keep_prob: 1.0
                         }

            similarity = self.sess.run(self.pos_similarity, feed_dict=feed_dict)

            for j in range(len(cand_sub_ids[i])):
                score[cand_sub_ids[i][j]] = similarity[j]

            scores.append(score)

        return scores


class SubTypeVec(object):
    def __init__(self, model_path):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(model_path+'.meta')
        self.saver.restore(self.sess, model_path)
        self.graph = tf.get_default_graph()

        # input
        self.question_ids = self.graph.get_tensor_by_name('question_ids:0')
        self.sequence_len = self.graph.get_tensor_by_name('sequence_len:0')
        self.sub_typevec = self.graph.get_tensor_by_name('sub_typevec:0')
        self.gru_keep_prob = self.graph.get_tensor_by_name('gru_kepp_prob:0')

        # output
        self.score = self.graph.get_tensor_by_name('sum_score:0')

    @staticmethod
    def get_typevec(type_ids):
        type_vec = np.zeros((len(type_ids), 500))
        for i in range(len(type_ids)):
            for type_id in type_ids[i]:
                type_vec[i][type_id] = 1
        return type_vec

    def infer(self, test_data):
        q_word_ids, q_seq_len, cand_sub_ids, can_sub_typeids = test_data

        scores = []
        data_size = q_word_ids.shape[0]
        for i in range(data_size):
            q_score = {}

            # compute score for each candidate subject
            dup_q_word_ids = np.tile(q_word_ids[i], (len(cand_sub_ids[i]), 1))
            dup_q_seq_len = np.tile(q_seq_len[i], len(cand_sub_ids[i]))

            feed_dict = {self.question_ids: dup_q_word_ids,
                         self.sequence_len: dup_q_seq_len,
                         self.sub_typevec: self.get_typevec(can_sub_typeids[i]),
                         self.gru_keep_prob: 1.0}
            result = self.sess.run(self.score, feed_dict=feed_dict)

            for j in range(len(cand_sub_ids[i])):
                q_score[cand_sub_ids[i][j]] = result[j]

            scores.append(q_score)

        return scores

