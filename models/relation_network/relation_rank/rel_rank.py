# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: rel_rnn.py

@time: 2018/4/6 10:53

@desc: learn a relation matching score function v(r, q) that measures the similarity between the question & the relation

"""

import os
import logging
import random
import numpy as np
import tensorflow as tf


class RelRank(object):
    def __init__(self, config, sess, word_embeddings):
        self.n_words = config.n_words
        self.n_relations = config.n_relations
        self.embedding_size = config.embedding_size
        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size
        self.margin = config.margin
        self.batch_size = config.batch_size
        self.max_sequence_len = config.max_sequence_len
        self.n_epoch = config.n_epoch
        self.early_stopping_step = config.early_stopping_step
        self.embedding_trainable = config.embedding_trainable   # whether to fine tune word embeddings
        
        self.model_path = config.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_name = config.model_name

        self.sess = sess

        self.loss = None
        self.pos_similarity = None
        self.neg_similarity = None
        self.train_step = None

        self.stopping_step = 0

        self.best_loss = 128000.0
        self.best_epoch = 0

        # define placeholder for input
        self.question_ids = tf.placeholder(tf.int32, [None, self.max_sequence_len], name='question_ids')
        self.sequence_len = tf.placeholder(tf.int32, name='sequence_len')
        self.pos_rel_ids = tf.placeholder(tf.int32, name='pos_rel_ids')     # positive relations
        self.neg_rel_ids = tf.placeholder(tf.int32, name='neg_rel_ids')     # negative relations
        self.gru_keep_prob = tf.placeholder(tf.float32, name='gru_keep_prob')

        # embedding layer
        self.word_embeddings = tf.Variable(initial_value=word_embeddings, name='word_embeddings', dtype=tf.float32,
                                           trainable=self.embedding_trainable)
        self.relation_embeddings = tf.Variable(tf.random_uniform([self.n_relations, self.embedding_size], -0.1, 0.1),
                                               name='relation_embeddings')

        self.pos_rel_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.pos_rel_ids)
        self.neg_rel_embed = tf.nn.embedding_lookup(self.relation_embeddings, self.neg_rel_ids)
        self.question_word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.question_ids)
        
        if not os.path.exists('log'):
            os.makedirs('log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line: %(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %b %d %H:%M:%S', filename='./log/train.log', filemode='a')
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

        # linear layer to project final hidden state of BiGRU to the same vector as entity embedding
        w_linear = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.embedding_size], stddev=0.01),
                               name='w_linear')
        b_linear = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name='b_linear')
        question_embed = tf.matmul(question_gru, w_linear) + b_linear

        # similarity layer
        self.pos_similarity = self.cosine_sim(question_embed, self.pos_rel_embed, name='pos_similarity')
        self.neg_similarity = self.cosine_sim(question_embed, self.neg_rel_embed, name='neg_similarity')

        # triplet loss
        # triplet_loss = tf.nn.relu(self.margin - self.pos_similarity + self.neg_similarity)
        # pos_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        # num_pos_triplets = tf.reduce_sum(pos_triplets)
        # self.loss = tf.reduce_sum(triplet_loss) / (num_pos_triplets + 1e-16)
        self.loss = tf.reduce_mean(tf.nn.relu(self.margin - self.pos_similarity + self.neg_similarity))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, train_data, valid_data, valid_data_metric):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=self.n_epoch)

        question_ids, sequence_len, pos_rel_ids, neg_rel_ids = train_data

        loops = int((len(question_ids) + self.batch_size - 1) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_loss = 0.0
            
            # shuffle data matters!
            shuffle_index = list(range(question_ids.shape[0]))
            random.shuffle(shuffle_index)
            
            for i in range(loops):
                feed_dict = {self.question_ids: question_ids[shuffle_index[i: i + self.batch_size]],
                             self.sequence_len: sequence_len[shuffle_index[i: i + self.batch_size]],
                             self.pos_rel_ids: pos_rel_ids[shuffle_index[i: i + self.batch_size]],
                             self.neg_rel_ids: neg_rel_ids[shuffle_index[i: i + self.batch_size]],
                             self.gru_keep_prob: 0.5}
                _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
                if loss == 0.0:
                    break
                avg_loss += loss

            avg_loss /= loops

            logging.info('%s %d train_loss: %f' % (self.model_name, epoch, avg_loss))
            print('%s %d train_loss: %f' % (self.model_name, epoch, avg_loss))

            saver.save(sess=self.sess, save_path=os.path.join(self.model_path, self.model_name), global_step=epoch)

            valid_loss = self.valid(valid_data)
            self.compute_accuracy(valid_data_metric, 'valid_acc')
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
        question_ids, sequence_len, pos_rel_ids, neg_rel_ids = valid_data
        # restrict the amount of data to be fed to avoid OOM error
        feed_dict = {self.question_ids: question_ids[:25600],
                     self.sequence_len: sequence_len[:25600],
                     self.pos_rel_ids: pos_rel_ids[:25600],
                     self.neg_rel_ids: neg_rel_ids[:25600],
                     self.gru_keep_prob: 1.0}
        loss = self.sess.run(self.loss, feed_dict=feed_dict)

        logging.info('%s valid_loss: %f' % (self.model_name, loss))
        print('%s valid_loss: %f' % (self.model_name, loss))

        return loss

    def compute_accuracy(self, data_metric, data_type):
        """"compute top k hits accuracy"""
        q_word_ids, q_seq_len, q_pos_rel_ids, cand_rel_ids = data_metric
        top1 = top3 = top5 = top10 = 0
        
        data_size = min(len(q_word_ids), 1000)
        for i in range(data_size):
            score = {}

            # compute score for each candidate relation
            q_word_ids_npy = np.zeros([1, 60])
            q_word_ids_npy[0, :len(q_word_ids[i])] = q_word_ids[i]
            mul_q_word_ids_npy = np.tile(q_word_ids_npy, (len(cand_rel_ids[i]), 1))

            mul_q_seq_len = np.tile(q_seq_len[i:i + 1], len(cand_rel_ids[i]))
            feed_dict = {self.question_ids: mul_q_word_ids_npy,
                         self.sequence_len: mul_q_seq_len,
                         self.pos_rel_ids: cand_rel_ids[i],
                         self.gru_keep_prob: 1.0
                         } 
            similarity = self.sess.run(self.pos_similarity, feed_dict=feed_dict)
        
            for j in range(len(cand_rel_ids[i])):
                score[cand_rel_ids[i][j]] = similarity[j]

            # rank by similarity score
            sorted_score = sorted(score.items(), key=lambda x:x[1], reverse=True)
            sorted_rel = [score[0] for score in sorted_score]

            if q_pos_rel_ids[i] in sorted_rel[:1]:
                top1 += 1
            if q_pos_rel_ids[i] in sorted_rel[:3]:
                top3 += 1
            if q_pos_rel_ids[i] in sorted_rel[:5]:
                top5 += 1
            if q_pos_rel_ids[i] in sorted_rel[:10]:
                top10 += 1
        print('%s %s: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' %
              (self.model_name, data_type, top1/data_size, top3/data_size,
               top5/data_size, top10/data_size))
        logging.info('%s %s: hits@1: %f hits@3: %f hits@5: %f hits@10: %f' %
              (self.model_name, data_type, top1/data_size, top3/data_size,
               top5/data_size, top10/data_size))

    @staticmethod
    def cosine_sim(a, b, name='cosine_sim'):
        a_norm = tf.nn.l2_normalize(a, axis=-1)
        b_norm = tf.nn.l2_normalize(b, axis=-1)
        return tf.reduce_sum(tf.multiply(a_norm, b_norm), axis=-1, name=name)

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

