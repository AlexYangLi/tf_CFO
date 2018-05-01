# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2018/4/6 10:20

@desc:

"""
import os
import numpy as np
import pickle
import tensorflow as tf
from bilstm_crf import BiLSTM_CRF
from gen_seqlabeling_data import read_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.app.flags

# hyper-parameter
flags.DEFINE_integer('n_words', 0, 'number of words')
flags.DEFINE_integer('embedding_size', 300, 'word embedding dimension')
flags.DEFINE_integer('hidden_size', 256, "lstm's hidden units")
flags.DEFINE_integer('n_layer', 2, 'number of lstm layer')
flags.DEFINE_integer('n_labels', 2, 'number of label')  # 2 kinds of label: whether is part of entity mention
flags.DEFINE_integer('max_sequence_len', 60, 'max number of words in one single question')
flags.DEFINE_integer('batch_size', 4096, 'batch size')
flags.DEFINE_integer('n_epoch', 20, 'max epoch to train')
flags.DEFINE_integer('early_stopping_step', 3, "if loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_float('stddev', 0.01, 'weight initialization stddev')
flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
flags.DEFINE_bool('bidirectional', True, 'options: lstm or bilstm')
flags.DEFINE_bool('add_crf', True, 'whether to add crf layer')
flags.DEFINE_string('train_data_path', '../../data/train.csv', 'path to train dataset')
flags.DEFINE_string('valid_data_path', '../../data/valid.csv', 'path to valid dataset')
flags.DEFINE_string('word2idx_path', '../../data/fb_word2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('word2vec_path', '../../data/fb_word2vec_300d.pkl', 'path to word2vec.pkl')

FLAGS = flags.FLAGS


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def pickle_save(obj, data_path):
    pickle.dump(obj, open(data_path, 'rb'))


def main(_):
    word2idx = pickle_load(FLAGS.word2idx_path)
    word2vec = pickle_load(FLAGS.word2vec_path)

    embedding_size = list(word2vec.values())[0].shape[0]
    word_embeddings = np.zeros([len(word2idx) + 1, embedding_size])  # index 0 for UNK token
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = word2vec[word]
    
    FLAGS.n_words = word_embeddings.shape[0]
    FLAGS.embedding_size = word_embeddings.shape[1]

    train_data = read_data(FLAGS.train_data_path, word2idx, FLAGS.max_sequence_len)
    valid_data = read_data(FLAGS.valid_data_path, word2idx, FLAGS.max_sequence_len)

    graph = tf.Graph()
    with tf.Session(graph=graph)as sess:
        model = BiLSTM_CRF(FLAGS, sess, word_embeddings)
        model.build_model()
        model.train(train_data,valid_data)


if __name__ == '__main__':
    tf.app.run()

