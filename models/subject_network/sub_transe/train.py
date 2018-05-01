# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2018/4/17 10:48

@desc:

"""

import os
import pickle
import tensorflow as tf
import numpy as np
from sub_transe import SubTransE
from gen_data import read_data

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


flags = tf.app.flags

flags.DEFINE_integer('n_words', 0, 'number of words')
flags.DEFINE_integer('n_subjects', 0, 'number of relations')
flags.DEFINE_integer('word_embed_size', 300, 'word embedding size')
flags.DEFINE_integer('sub_embed_size', 50, 'subject embedding size')
flags.DEFINE_integer('n_layer', 2, 'number of gru layer')
flags.DEFINE_integer('hidden_size', 256, 'num hidden units of gru')
flags.DEFINE_float('margin', 1.5, 'margin in margin-based loss function')
flags.DEFINE_integer('batch_size', 4096, 'batch size')
flags.DEFINE_integer('max_sequence_len', 60, 'max number of words in one single question')
flags.DEFINE_integer('neg_sample_size', 50, 'negative samples\' size')
flags.DEFINE_integer('n_epoch', 50, 'number of epochs')
flags.DEFINE_integer('early_stopping_step', 3, "f loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_bool('word_embed_trainable', True, 'whether word embedding can be tune during training')
flags.DEFINE_bool('sub_embed_trainable', False, 'whether subject embedding can be tune during training')
flags.DEFINE_string('train_data_path', '../../../data/train.csv', 'path to train dataset')
flags.DEFINE_string('valid_data_path', '../../../data/valid.csv', 'path to valid dataset')
flags.DEFINE_string('word2idx_path', '../../../data/fb_word2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('subject2idx_path', '../../../data/FB5M_subject2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('word2vec_path', '../../../data/fb_word2vec_300d.pkl', 'path to word2vec.pkl')
flags.DEFINE_string('subject_embedding_path', '../../../data/sub_transe_embed.npy', 'path to subject embeddings')
FLAGS = flags.FLAGS


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def main(_):
    word2idx = pickle_load(FLAGS.word2idx_path)
    subject2idx = pickle_load(FLAGS.subject2idx_path)
    word2vec = pickle_load(FLAGS.word2vec_path)
    subject_embeddings = np.load(FLAGS.subject_embedding_path)
 
    word_embed_size = list(word2vec.values())[0].shape[0]
    word_embeddings = np.zeros([len(word2idx) + 1, word_embed_size])  # index 0 for UNK token
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = word2vec[word]

    FLAGS.n_words = word_embeddings.shape[0]
    FLAGS.n_subjects = len(subject2idx.keys())
    FLAGS.word_embed_size = word_embeddings.shape[1]
    FLAGS.sub_embed_size = subject_embeddings.shape[1]

    train_data, _ = read_data(FLAGS.train_data_path, word2idx, subject2idx, FLAGS.max_sequence_len, FLAGS.neg_sample_size)
    valid_data, valid_data_metric = read_data(FLAGS.valid_data_path, word2idx, subject2idx, FLAGS.max_sequence_len)

    graph = tf.Graph()
    with tf.Session(graph=graph)as sess:
        model = SubTransE(FLAGS, sess, word_embeddings, subject_embeddings)
        model.build_model()
        model.train(train_data, valid_data, valid_data_metric)


if __name__ == '__main__':
    tf.app.run()

