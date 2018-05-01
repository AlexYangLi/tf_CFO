# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2018/4/18 9:03

@desc:

"""

import os
import pickle
import tensorflow as tf
import numpy as np
from sub_typevec import SubTypeVec
from gen_data import read_data

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


flags = tf.app.flags

flags.DEFINE_integer('n_words', 0, 'number of words')
flags.DEFINE_integer('n_subtype', 0, 'subject type vector dimension')
flags.DEFINE_integer('embedding_size', 300, 'word embedding size')
flags.DEFINE_integer('n_layer', 2, 'number of gru layer')
flags.DEFINE_integer('hidden_size', 256, 'num hidden units of gru')
flags.DEFINE_integer('batch_size', 4096, 'batch size')
flags.DEFINE_integer('max_sequence_len', 60, 'max number of words in one single question')
flags.DEFINE_integer('n_epoch', 100, 'number of epochs')
flags.DEFINE_integer('early_stopping_step', 3, "f loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_bool('embedding_trainable', False, 'whether word embedding can be tune during training')
flags.DEFINE_string('train_data_path', '../../../../data/train.csv', 'path to train dataset')
flags.DEFINE_string('valid_data_path', '../../../../data/valid.csv', 'path to valid dataset')
flags.DEFINE_string('word2idx_path', '../../../../data/fb_word2idx.pkl', 'path to word2idx.pkl')
flags.DEFINE_string('word2vec_path', '../../../../data/fb_word2vec_300d.pkl', 'path to word2vec.pkl')
flags.DEFINE_string('idx2subject_path', '../../../../data/FB5M_idx2subject.pkl', 'path to idx2subject.pkl')
flags.DEFINE_string('subject2idx_path', '../../../../data/FB5M_subject2idx.pkl', 'path to subject2idx.pkl')
flags.DEFINE_string('subject2type_path', '../../../../data/trim_subject2type.pkl', 'path to subject2type.pkl')
flags.DEFINE_string('type2idx_path', '../../../../data/FB5M_type2idx.pkl', 'path to type2idx.pkl')
flags.DEFINE_string('model_path', './save_model', 'path to type2idx.pkl')
flags.DEFINE_string('model_name', 'sub_typevec', 'model name')
FLAGS = flags.FLAGS


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def main(_):
    word2idx = pickle_load(FLAGS.word2idx_path)
    word2vec = pickle_load(FLAGS.word2vec_path)
    idx2subject = pickle_load(FLAGS.idx2subject_path)
    subject2idx = pickle_load(FLAGS.subject2idx_path)
    subject2type = pickle_load(FLAGS.subject2type_path)
    type2idx = pickle_load(FLAGS.type2idx_path)

    embedding_size = list(word2vec.values())[0].shape[0]
    word_embeddings = np.zeros([len(word2idx) + 1, embedding_size])  # index 0 for UNK token
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = word2vec[word]

    FLAGS.n_words = word_embeddings.shape[0]
    FLAGS.embedding_size = word_embeddings.shape[1]
    FLAGS.n_subtype = len(type2idx)

    print(FLAGS.n_subtype)
    print(FLAGS.embedding_size)

    train_data = read_data(FLAGS.train_data_path, word2idx, idx2subject, subject2idx, subject2type, type2idx, FLAGS.max_sequence_len)
    valid_data = read_data(FLAGS.valid_data_path, word2idx, idx2subject, subject2idx, subject2type, type2idx, FLAGS.max_sequence_len)

    graph = tf.Graph()
    with tf.Session(graph=graph)as sess:
        model = SubTypeVec(FLAGS, sess, word_embeddings, type2idx)
        model.build_model()
        model.train(train_data, valid_data)


if __name__ == '__main__':
    tf.app.run()
