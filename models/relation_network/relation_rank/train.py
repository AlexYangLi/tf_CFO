
# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2018/4/11 22:57

@desc:

"""

import os
import pickle
import tensorflow as tf
import numpy as np
from rel_rank import RelRank
from gen_relrank_data import read_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


flags = tf.app.flags

flags.DEFINE_integer('n_words', 0, 'number of words')
flags.DEFINE_integer('n_relations', 0, 'number of relations')
flags.DEFINE_integer('embedding_size', 300, 'word embedding & relation embedding')
flags.DEFINE_integer('n_layer', 2, 'number of gru layer')
flags.DEFINE_integer('hidden_size', 256, 'num hidden units of gru')
flags.DEFINE_float('margin', 1.0, 'margin in margin-based loss function')
flags.DEFINE_integer('batch_size', 4096, 'batch size')
flags.DEFINE_integer('max_sequence_len', 60, 'max number of words in one single question')
flags.DEFINE_integer('neg_sample_size', 50, 'negative samples\' size')
flags.DEFINE_integer('n_epoch', 50, 'number of epochs')
flags.DEFINE_integer('early_stopping_step', 3, "f loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_bool('embedding_trainable', False, 'whether word embedding can be tune during training')
flags.DEFINE_string('train_data_path', '../../../data/train.csv', 'path to train dataset')
flags.DEFINE_string('valid_data_path', '../../../data/valid.csv', 'path to valid dataset')
flags.DEFINE_string('word2idx_path', '../../../data/fb_word2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('relation2idx_path', '../../../data/FB5M_relation2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('word2vec_path', '../../../data/fb_word2vec_300d.pkl', 'path to word2vec.pkl')
flags.DEFINE_string('subject2rel_path', '../../../data/FB5M_subject2rel.pkl', 'path to freebase triple')
flags.DEFINE_string('model_path', './save_model', 'path to save model')
flags.DEFINE_string('model_name', 'RelRank_random', 'model name')
FLAGS = flags.FLAGS


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))

def main(_):
    word2idx = pickle_load(FLAGS.word2idx_path)
    relation2idx = pickle_load(FLAGS.relation2idx_path)
    word2vec = pickle_load(FLAGS.word2vec_path)
    subject2rel = pickle_load(FLAGS.subject2rel_path)

    embedding_size = list(word2vec.values())[0].shape[0]
    word_embeddings = np.zeros([len(word2idx) + 1, embedding_size])  # index 0 for UNK token
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = word2vec[word]

    FLAGS.n_words = word_embeddings.shape[0]
    FLAGS.n_relations = len(relation2idx.keys())
    FLAGS.embedding_size = word_embeddings.shape[1]

    train_data, _ = read_data(FLAGS.train_data_path, subject2rel, word2idx, relation2idx, FLAGS.max_sequence_len, FLAGS.neg_sample_size, restrict=False)
    valid_data, valid_data_metric = read_data(FLAGS.valid_data_path, subject2rel, word2idx, relation2idx, FLAGS.max_sequence_len, restrict=False)

    graph = tf.Graph()
    with tf.Session(graph=graph)as sess:
        model = RelRank(FLAGS, sess, word_embeddings)
        model.build_model()
        model.train(train_data, valid_data, valid_data_metric)


if __name__ == '__main__':
    tf.app.run()
