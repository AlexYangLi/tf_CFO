
# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: sub_typevec.py

@time: 2018/4/6 22:10

@desc: learn a K binary classifiers which can correctly classify a question into the types that
       associated with the target subject

"""

import os
import logging
import numpy as np
import tensorflow as tf


class SubTypeVec(object):
    def __init__(self, config, sess, word_embeddings, type2idx):
        self.n_words = config.n_words
        self.n_subtype = config.n_subtype
        self.embedding_size = config.embedding_size
        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.max_sequence_len = config.max_sequence_len
        self.n_epoch = config.n_epoch
        self.early_stopping_step = config.early_stopping_step
        self.embedding_trainable = config.embedding_trainable   # whether to fine tune word embeddings
        self.type2idx = type2idx

        self.model_path = config.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_name = config.model_name
        self.log_path = './log'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.sess = sess

        self.loss = None
        self.train_step = None
        self.mean_score = None
        self.sum_score = None

        self.stopping_step = 0
        self.best_loss = 128.0
        self.best_epoch = 0

        # define placeholder for input
        self.question_ids = tf.placeholder(tf.int32, [None, self.max_sequence_len], name='question_ids')
        self.sequence_len = tf.placeholder(tf.int32, name='sequence_len')
        """
        encode the subject as a vector (bag) of types. Each dimension of a type vector is either 1 or 0, indicating 
        whether the subject is associated with a specific type or not.
        """
        self.sub_typevec = tf.placeholder(tf.float32, [None, self.n_subtype], name='sub_typevec')
        self.gru_keep_prob = tf.placeholder(tf.float32, name='gru_kepp_prob')

        # embedding layer
        self.word_embeddings = tf.Variable(initial_value=word_embeddings, name='word_embeddings',
                                           dtype=tf.float32, trainable=self.embedding_trainable)
        self.question_word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.question_ids)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line: %(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %b %d %H:%M:%S', filename=os.path.join(self.log_path, 'train.log')
                            , filemode='a')
        logging.info(config)

    def single_gru_cell(self):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.hidden_size))

    def multi_gru_cell(self):
        stack_gru = [self.single_gru_cell() for _ in range(self.n_layer)]
        return tf.nn.rnn_cell.MultiRNNCell(stack_gru)

    def build_model(self):
        # bigru layer
        cell_fw = self.multi_gru_cell()
        cell_bw = self.multi_gru_cell()
        _, ((_, fw_state), (_, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=self.question_word_embed,
                                                                            sequence_length=self.sequence_len,
                                                                            dtype=tf.float32)
        question_gru = tf.concat([fw_state, bw_state], axis=-1)     # [batch_size, 2*hidden_size]

        # linear layer to project final hidden state of BiGRU to the same vector as subject type vector
        w_linear = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.n_subtype], stddev=0.01),
                               name='w_linear')
        b_linear = tf.Variable(tf.constant(0.1, shape=[self.n_subtype]), name='b_linear')
        question_embed = tf.matmul(question_gru, w_linear) + b_linear

        # type wise binary cross-entropy loss
        type_wise_logistic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=question_embed,
                                                                          labels=self.sub_typevec)
        self.loss = tf.reduce_mean(tf.reduce_sum(type_wise_logistic_loss, axis=-1))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        question_embed_sigmoid = tf.nn.sigmoid(question_embed, name='question_embed_sigmoid')
        self.sum_score = tf.reduce_sum(tf.multiply(question_embed_sigmoid, self.sub_typevec), axis=-1,
                                       name='sum_score')

        type_count = tf.reduce_sum(self.sub_typevec, axis=-1)
        self.mean_score = tf.div(self.sum_score, type_count, name='mean_score')

    def train(self, train_data, valid_data):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        question_ids, sequence_len, sub_typeids, _, _ = train_data

        loops = int((len(question_ids) + self.batch_size - 1) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_loss = 0.0
            for i in range(loops):
                feed_dict = {self.question_ids: question_ids[i: i+self.batch_size],
                             self.sequence_len: sequence_len[i: i+self.batch_size],
                             self.sub_typevec: self.get_typevec(sub_typeids[i: i+self.batch_size]),
                             self.gru_keep_prob: 0.5}
                _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
                # print('%s %d train_loss: %f' % (self.model_name, i, loss))
                avg_loss += loss

            avg_loss /= loops
            logging.info('%s %d train_loss: %f' % (self.model_name, epoch, avg_loss))

            print('%s %d train_loss: %f' % (self.model_name, epoch, avg_loss))

            saver.save(sess=self.sess, save_path=os.path.join(self.model_path, self.model_name), global_step=epoch)

            valid_loss = self.valid(valid_data)
            self.compute_accuracy(valid_data, 'sum')
            self.compute_accuracy(valid_data, 'mean')

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                print('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                logging.info('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                break

        print('%s best epoch: %d, best loss: %f' % (self.model_name, self.best_epoch, self.best_loss))
        logging.info('%s best epoch: %d, best loss: %f' % (self.model_name, self.best_epoch, self.best_loss))

        self.only_save_best_epoch(self.model_path, self.model_name, self.best_epoch)

    def valid(self, valid_data):
        question_ids, sequence_len, sub_typeids, _, _ = valid_data
        feed_dict = {self.question_ids: question_ids,
                     self.sequence_len: sequence_len,
                     self.sub_typevec: self.get_typevec(sub_typeids),
                     self.gru_keep_prob: 1.0}
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        logging.info('%s valid_loss: %f' % (self.model_name, loss))

        print('%s valid loss: %f' % (self.model_name, loss))

        return loss

    def compute_accuracy(self, data_metric, score_func):
        """compute top k hits accuracy"""
        question_ids, sequence_len, pos_sub_typeids, cand_subs, can_sub_typeids = data_metric
        top1 = top3 = top5 = top10 = 0

        data_size = min(len(question_ids), 1000)
        for i in range(data_size):
            score = {}

            num_cand = len(cand_subs[i])
            pos_sub_idx = cand_subs[i][0]    # index 0 will always be positive

            # compute score for each candidate subject
            mul_q_word_ids = np.tile(question_ids[i], (num_cand, 1))
            mul_q_seq_len = np.tile(sequence_len[i], num_cand)
            feed_dict = {self.question_ids: mul_q_word_ids,
                         self.sequence_len: mul_q_seq_len,
                         self.sub_typevec: self.get_typevec(can_sub_typeids[i]),
                         self.gru_keep_prob: 1.0}
            if score_func == 'sum':
                result = self.sess.run(self.sum_score, feed_dict=feed_dict)
            else:
                result = self.sess.run(self.mean_score, feed_dict=feed_dict)
            
            for j in range(num_cand):
                score[cand_subs[i][j]] = result[j]

            # rank by score
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
            sorted_sub = [score[0] for score in sorted_score]

            if pos_sub_idx in sorted_sub[:1]:
                top1 += 1
            if pos_sub_idx in sorted_sub[:3]:
                top3 += 1
            if pos_sub_idx in sorted_sub[:5]:
                top5 += 1
            if pos_sub_idx in sorted_sub[:10]:
                top10 += 1

        print('%s %s score rank: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' %
              (self.model_name, score_func, top1 / data_size, top3 / data_size,
               top5 / data_size, top10 / data_size))
        logging.info('%s %s score rank: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' %
                     (self.model_name, score_func, top1 / data_size, top3 / data_size,
                      top5 / data_size, top10 / data_size))

    def get_typevec(self, type_ids):
        type_vec = np.zeros((len(type_ids), self.n_subtype))
        for i in range(len(type_ids)):
            for type_id in type_ids[i]:
                type_vec[i][type_id] = 1
        return type_vec

    @staticmethod
    def only_save_best_epoch(model_path, model_name, best_epoch):
        data_suffix = '.data-00000-of-00001'
        data_name = model_name + '-' + str(best_epoch) + data_suffix

        index_suffix = '.index'
        index_name = model_name + '-' + str(best_epoch) + index_suffix

        meta_suffix = '.meta'
        meta_name = model_name + '-' + str(best_epoch) + meta_suffix

        for file in os.listdir(model_path):
            if file.startswith(model_name):
                if file == data_name or file == index_name or file == meta_name:
                    continue
                elif file.endswith(data_suffix) or file.endswith(index_suffix) or file.endswith(meta_suffix):
                    os.remove(os.path.join(model_path, file))

