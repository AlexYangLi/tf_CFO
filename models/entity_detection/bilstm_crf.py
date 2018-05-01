# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: bilstm_crf.py

@time: 2018/4/5 16:59

@desc: bilstm + crf for sequence labeling

"""

import os
import logging
import tensorflow as tf


class BiLSTM_CRF(object):
    def __init__(self, config, sess, word_embeddings):
        self.n_words = config.n_words
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer
        self.n_labels = config.n_labels
        self.max_sequence_len = config.max_sequence_len   # max length of one question
        self.early_stopping_step = config.early_stopping_step
        self.stddev = config.stddev
        self.l2_reg = config.l2_reg
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.bidirectional = config.bidirectional   # options: lstm or bilstm
        self.add_crf = config.add_crf   # whether add crf layer

        self.model_path = './save_model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_name = ('bilstm' if self.bidirectional else 'lstm') + ('_crf' if self.add_crf else '')

        self.sess = sess
        self.loss = None
        self.predict = None
        # self.
        self.macro_precision = None
        self.macro_recall = None
        self.macro_f1 = None
        self.train_step = None

        self.best_epoch = 0
        self.best_loss = 128.0
        self.best_precision = 0.0
        self.best_recall = 0.0
        self.best_f1 = 0.0
        self.stopping_step = 0

        # define placeholder for input
        self.question_word_ids = tf.placeholder(tf.int32, [None, self.max_sequence_len], name='question_word_ids')
        self.sequence_len = tf.placeholder(tf.int32, name='sequence_len')
        self.labels = tf.placeholder(tf.int32, [None, self.max_sequence_len], name='label')
        self.rnn_keep_prob = tf.placeholder(tf.float32, name='rnn_keep_prob')

        # embedding layer
        # cast embedding variable to tf.float32 to avoid causing the concat between tf.float32 &
        # tf.float64 error in bidrectional_dynamic_rnn function.
        # reference: https://stackoverflow.com/questions/43452873/bidirectional-dynamic-rnn-function-in-tensorflow-1-0
        self.word_embeddings = tf.Variable(initial_value=word_embeddings, name='word_embeddings', dtype=tf.float32,
                                           trainable=False)    
        self.question_embed = tf.nn.embedding_lookup(self.word_embeddings, self.question_word_ids)

        # log	
        if not os.path.exists('log'):
            os.makedirs('log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line: %(lineno)d)] %(levelname)s %(message)s',
                            datefmt='%a, %b %d %Y %H:%M:%s', filename='./log/trainging.log', filemode='a')

    def single_lstm_cell(self):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True),
                                             output_keep_prob=self.rnn_keep_prob)

    def multi_lstm_cell(self):
        stack_lstm = [self.single_lstm_cell() for _ in range(self.n_layer)]
        return tf.nn.rnn_cell.MultiRNNCell(stack_lstm, state_is_tuple=True)

    def lstm_model(self):
        with tf.variable_scope('question_lstm') as scope:
            cell = self.single_lstm_cell() if self.n_layer > 1 else self.multi_lstm_cell()
            outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=self.question_embed,
                                           sequence_length=self.sequence_len,
                                           dtype=tf.float32)
        return outputs  # [batch_size, sequence_len, hidden_size]

    def bilstm_model(self):
        with tf.variable_scope('question_bilstm') as scope:
            cell_fw = self.single_lstm_cell() if self.n_layer > 1 else self.multi_lstm_cell()
            cell_bw = self.single_lstm_cell() if self.n_layer > 1 else self.multi_lstm_cell()

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                         cell_bw=cell_bw,
                                                         inputs=self.question_embed,
                                                         sequence_length=self.sequence_len,
                                                         dtype=tf.float32)
        return tf.concat(outputs, axis=-1)    # [batch_size, sequence_len, 2*hidden_size]

    def build_model(self):
        # lstm layer
        if self.bidirectional:
            question_encoded = self.bilstm_model()
            question_encoded_reshape = tf.reshape(question_encoded, [-1, 2*self.hidden_size])
            w_softmax = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.n_labels], stddev=self.stddev),
                                    name='w_softmax')
        else:
            question_encoded = self.lstm_model()
            question_encoded_reshape = tf.reshape(question_encoded, [-1, self.hidden_size])
            w_softmax = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_labels], stddev=self.stddev),
                                    name='w_softmax')
        b_softmax = tf.Variable(tf.constant(0.1, shape=[self.n_labels]), name='b_softmax')
       
        network_scores = tf.matmul(question_encoded_reshape, w_softmax) + b_softmax    # [batch_size*sequence_len, n_labels]
        network_scores = tf.reshape(network_scores, [-1, self.max_sequence_len, self.n_labels], name='network_scores')

        # crf layer
        if self.add_crf:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=network_scores,
                                                                                  tag_indices=self.labels,
                                                                                  sequence_lengths= self.sequence_len)
            # print(transition_params)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.predict, _ = tf.contrib.crf.crf_decode(potentials=network_scores,
                                                        transition_params=transition_params,
                                                        sequence_length=self.sequence_len)  # [batch_size, sequence_len]
            # print(self.predict)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=network_scores, labels=self.labels)
            mask = tf.sequence_mask(lengths=self.sequence_len, maxlen=self.max_sequence_len)
            # need to explicitly declare the shape to mask
            # otherwise a Value Error is raised when using tf.boolean_mask: Number of mask dimensions must be specified
            # , even if some dimensions are None.
            # refer to https://stackoverflow.com/questions/44010410/tf-boolean-mask-mask-dimension-must-be-specified
            mask.set_shape([None, self.max_sequence_len])
            mask_cross_entropy = tf.boolean_mask(cross_entropy, mask)    # one dimension

            self.loss = tf.reduce_mean(mask_cross_entropy)
            self.predict = tf.multiply(tf.argmax(network_scores, axis=-1), tf.cast(mask, tf.int64), name='predict')

        self.macro_precision, self.macro_recall, self.macro_f1 = self.tf_confusion_metrics(self.predict, self.labels,
                                                                                           self.sequence_len,
                                                                                           self.max_sequence_len)

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, train_data, valid_data):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=self.n_epoch)

        question_word_ids, sequence_len, labels = train_data

        loops = int((len(question_word_ids) + self.batch_size - 1) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_loss = avg_precision = avg_recall = avg_f1 = 0.0
            for i in range(loops):
                feed_dict = {self.question_word_ids: question_word_ids[i: i+self.batch_size],
                             self.sequence_len: sequence_len[i: i+self.batch_size],
                             self.labels: labels[i: i+self.batch_size],
                             self.rnn_keep_prob: 0.5}
                _, loss, precision, recall, f1 = self.sess.run([self.train_step, self.loss, self.macro_precision,
                                                                self.macro_recall, self.macro_f1], feed_dict=feed_dict)

                # print('%s %d batch_loss: %f batch_precision: %f batch_recall: %f batch_f1: %f' %
                #      (self.model_name, i, loss, precision, recall, f1))
                
                avg_loss += loss; avg_precision += precision; avg_recall += recall; avg_f1 += f1
            
            avg_loss /= loops; avg_precision /= loops; avg_recall /= loops; avg_f1 /= loops
            
            logging.info('%s %d train_loss: %f train_precision: %f train_recall: %f train_f1: %f' %
                         (self.model_name, epoch, avg_loss, avg_precision, avg_recall, avg_f1))

            print('%s %d train_loss: %f train_precision: %f train_recall: %f train_f1: %f' %
                  (self.model_name, epoch, avg_loss, avg_precision, avg_recall, avg_f1))

            saver.save(sess=self.sess, save_path=os.path.join(self.model_path, self.model_name), global_step=epoch)

            valid_loss, valid_precision, valid_recall, valid_f1 = self.valid(valid_data)
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss; self.best_precision = valid_precision
                self.best_recall = valid_recall; self.best_f1 = valid_f1
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                print('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                logging.info('%s early stopping is trigger at epoch: %d' % (self.model_name, epoch))
                break

        print('%s best epoch: %d, best loss: %f best precision: %f best recall: %f best f1: %f ' %
              (self.model_name, self.best_epoch, self.best_loss, self.best_precision, self.best_recall, self.best_f1))
        logging.info('%s best epoch: %d, best loss: %f best precision: %f best recall: %f best f1: %f ' %
                     (self.model_name, self.best_epoch, self.best_loss, self.best_precision, self.best_recall, self.best_f1))

        self.only_save_best_epoch(self.model_path, self.model_name, self.best_epoch)

    def valid(self, valid_data):
        question_word_ids, sequence_len, labels = valid_data
        feed_dict = {self.question_word_ids: question_word_ids,
                     self.sequence_len: sequence_len,
                     self.labels: labels,
                     self.rnn_keep_prob: 1.0}
        loss, precision, recall, f1 = self.sess.run([self.loss, self.macro_precision, self.macro_recall, self.macro_f1],
                                                    feed_dict=feed_dict)
        logging.info('%s valid_loss: %f valid_precision: %f valid_recall: %f valid_f1: %f' %
                     (self.model_name, loss, precision, recall, f1))

        print('%s valid_loss: %f valid_precision: %f valid_recall: %f valid_f1: %f' %
                     (self.model_name, loss, precision, recall, f1))

        return loss, precision, recall, f1


    @staticmethod
    #  refer to https://gist.github.com/Mistobaan/337222ac3acbfc00bdac
    def tf_confusion_metrics(predictions, actuals, item_length, max_item_length):
        """
        compute macro precision, macro recall & average f1
        :param predictions: [batch_size, item_size]. For each item in each sample, we do the classification.
                            predictions[i][j] (jth item in ith sample) is either 1 or 0, indicating positive or negative
        :param actuals: [batch_size, item_size]
        :param item_length: [batch_size, ]. determine exactly how many items in each sample (no larger than item_size)
        :return:
        """

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        mask = tf.sequence_mask(lengths=item_length, maxlen=max_item_length)    # boolean tensor:[batch_size, item_size]

        # True Positive [batch_size,]
        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictions, ones_like_predictions)
                    ),
                    mask
                ),
                tf.float32
            ),
            axis=-1
        )

        # False Positive [batch_size, ]
        fp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictions, ones_like_predictions)
                    ),
                    mask
                ),
                tf.float32
            ),
            axis=-1
        )

        # True Negative
        tn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictions, zeros_like_predictions)
                    ),
                    mask
                ),
                tf.float32
            ),
            axis=-1
        )

        # False Negative
        fn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictions, zeros_like_predictions)
                    ),
                    mask
                ),
                tf.float32
            ),
            axis=-1
        )

        precision = tp / (tp + fp + 0.000001)
        recall = tp / (tp + fn + 0.000001)
        f1_score = (2 * precision * recall) / (precision + recall + 0.000001)

        macro_precision = tf.reduce_mean(precision, name='macro_precision')
        macro_recall = tf.reduce_mean(recall, name='macro_recall')
        macro_f1 = tf.reduce_mean(f1_score, name='macro_f1')

        return macro_precision, macro_recall, macro_f1

    def only_save_best_epoch(self, model_path, model_name, best_epoch):
        data_suffix = '.data-00000-of-00001'
        data_name = model_name + '-' + str(best_epoch) + data_suffix

        index_suffix = '.index'
        index_name = model_name + '-' + str(best_epoch) + index_suffix

        meta_suffix = '.meta'
        meta_name = model_name + '-' + str(best_epoch) + meta_suffix

        for file in os.listdir(model_path):
            if file.startswith(model_name):
                if not self.add_crf and file.startswith(model_name + '_crf'):
                    continue
                if file == data_name or file==index_name or file==meta_name:
                    continue
                elif file.endswith(data_suffix) or file.endswith(index_suffix) or file.endswith(meta_suffix):
                    os.remove(os.path.join(model_path, file))

