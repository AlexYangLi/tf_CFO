
# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: rel_classify.py

@time: 2018/4/13 10:04

@desc:

"""


import os
import logging
import tensorflow as tf


class RelClassify(object):
    def __init__(self, config, sess, word_embeddings):
        self.n_words = config.n_words
        self.n_relations = config.n_relations
        self.embedding_size = config.embedding_size
        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.max_sequence_len = config.max_sequence_len
        self.n_epoch = config.n_epoch
        self.early_stopping_step = config.early_stopping_step
        self.embedding_trainable = config.embedding_trainable   # whether to fine tune word embeddings

        self.model_path = './save_model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_name = 'RelationClassify'

        self.sess = sess

        self.loss = None
        self.predict = None
        self.accuracy = None
        self.train_step = None

        self.stopping_step = 0
        self.best_loss = 128.0
        self.best_acc = 0.0
        self.best_epoch = 0

        # define placeholder for input
        self.question_ids = tf.placeholder(tf.int32, [None, self.max_sequence_len], name='question_ids')
        self.sequence_len = tf.placeholder(tf.int32, name='sequence_len')
        self.pos_rel_ids = tf.placeholder(tf.int64, name='pos_rel_ids')     # positive relations
        self.gru_keep_prob = tf.placeholder(tf.float32, name='gru_keep_prob')

        # embedding layer
        self.word_embeddings = tf.Variable(initial_value=word_embeddings, name='word_embeddings', dtype=tf.float32,
                                           trainable=self.embedding_trainable)
        self.question_word_embed = tf.nn.embedding_lookup(self.word_embeddings, self.question_ids)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line: %(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %b %d %H:%M:%S', filename='./log/train.log', filemode='a')

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

        # 2 full-connected layer to project final hidden state of BiGRU to the same vector as entity embedding
        w_linear = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.n_relations], stddev=0.01),
                               name='w_linear')
        b_linear = tf.Variable(tf.constant(0.1, shape=[self.n_relations]), name='b_linear')
        #linear_out = tf.nn.relu(tf.matmul(question_gru, w_linear) + b_linear)
        
        #w_final = tf.Variable(tf.truncated_normal([128, self.n_relations], stddev=0.01),
        #                      name='w_final')
        #b_final = tf.Variable(tf.constant(0.1, shape=[self.n_relations]), name='b_final')
        question_embed = tf.matmul(question_gru, w_linear) + b_linear

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=question_embed, labels=self.pos_rel_ids)
        self.loss = tf.reduce_mean(cross_entropy)
        self.predict = tf.argmax(question_embed, axis=-1, name='predict')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.pos_rel_ids), tf.float32), name='accuracy')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, train_data, valid_data):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=self.n_epoch)

        question_ids, sequence_len, pos_rel_ids = train_data

        loops = int((len(question_ids) + self.batch_size - 1) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_loss = avg_acc = 0.0
            for i in range(loops):
                feed_dict = {self.question_ids: question_ids[i: i + self.batch_size],
                             self.sequence_len: sequence_len[i: i + self.batch_size],
                             self.pos_rel_ids: pos_rel_ids[i: i + self.batch_size],
                             self.gru_keep_prob: 0.5}
                _, loss, accuracy = self.sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)

                #print('%s %d train_loss: %f train_acc: %f' % (self.model_name, i, loss, accuracy))

                avg_loss += loss
                avg_acc += accuracy

            avg_loss /= loops
            avg_acc /= loops

            logging.info('%s %d train_loss: %f train_acc: %f' % (self.model_name, epoch, avg_loss, avg_acc))
            print('%s %d train_loss: %f train_acc: %f' % (self.model_name, epoch, avg_loss, avg_acc))

            saver.save(sess=self.sess, save_path=os.path.join(self.model_path, self.model_name), global_step=epoch)

            valid_loss, valid_acc  = self.valid(valid_data)
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_acc = valid_acc
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                print('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                logging.info('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                break

        print('%s best epoch: %d, best loss: %f best acc: %f' % (self.model_name, self.best_epoch, self.best_loss, self.best_acc))
        logging.info('%s best epoch: %d, best loss: %f best acc: %f' % (self.model_name, self.best_epoch, self.best_loss, self.best_acc))

        self.only_save_best_epoch(self.model_path, self.model_name, self.best_epoch)

    def valid(self, valid_data):
        question_ids, sequence_len, pos_rel_ids = valid_data
        feed_dict = {self.question_ids: question_ids,
                     self.sequence_len: sequence_len,
                     self.pos_rel_ids: pos_rel_ids,
                     self.gru_keep_prob: 1.0}
        loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        logging.info('%s valid_loss: %f valid_acc: %f' % (self.model_name, loss, accuracy))
        print('%s valid_loss: %f valid_acc: %f' % (self.model_name, loss, accuracy))

        return loss, accuracy

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


