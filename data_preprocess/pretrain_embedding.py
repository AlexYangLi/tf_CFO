# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: pretrain_embedding.py

@time: 2018/4/7 21:02

@desc: use gensim to pre_train word embeddings

"""
import os
import sys
import logging
import nltk
from gensim.models import Word2Vec
from utils import pickle_save


def train(data_path, save_dir):
    sentences = []
    data_files = [os.path.join(os.path.dirname(data_path), file) for file in os.listdir(data_path)]
    for data_file in data_files:
        with open(data_file, 'r')as reader:
            for line in reader:
                question = line.strip().split('\t')[-1].lower()
                sentences.append(nltk.word_tokenize(question))

    model = Word2Vec(sentences, size=300, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])

    embeddings_index = {}
    for item in d:
        embeddings_index[item] = weights[d[item], :]
    pickle_save(embeddings_index, os.path.join(save_dir, 'fb_word2vec_300d.pkl'))

    word2idx = {}
    for idx, word in enumerate(embeddings_index.keys()):
        word2idx[word] = idx+1  # index 0 refers to unknown token
    pickle_save(word2idx, os.path.join(save_dir, 'fb_word2idx.pkl'))
   
    char2idx = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
                'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
                't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1':28,
                '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36}
    pickle_save(char2idx, os.path.join(save_dir, 'fb_char2idx.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    train(sys.argv[1], sys.argv[2])

