# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: test_vocab.py

@time: 2018/4/8 21:24

@desc:

"""

import os
import nltk
from utils import pickle_load

word2idx = pickle_load('../data/fb_word2idx.pkl')
vocab = word2idx.keys()

data_path = '../raw_data/SimpleQuestions_v2/dataset/'
data_files = [os.path.join(os.path.dirname(data_path), file) for file in os.listdir(data_path)]
for data_file in data_files:
    with open(data_file, 'r')as reader:
        good = 0
        for index, line in enumerate(reader):
            question = line.strip().split('\t')[-1]
            tokens = nltk.word_tokenize(question.lower())
            in_tokens = [word for word in tokens if word in vocab]
            ratio = len(in_tokens) / len(tokens)
            if ratio < 1.0:
                print(question)
            else:
                good += 1
        print(good, index+1)
