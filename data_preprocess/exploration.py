# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: exploration.py

@time: 2018/4/9 8:41

@desc:

"""

import numpy as np
import pandas as pd
from utils import pickle_load

word2idx = pickle_load('../data/fb_word2idx.pkl')
word_len = []
for word in word2idx.keys():
    word_len.append(len(word))

print('max word len: ', np.max(word_len))
print('min word len: ', np.min(word_len))
print('avg word len: ', np.average(word_len))
print('median word len: ', np.median(word_len))

for i in range(int(np.median(word_len)), int(np.max(word_len)), 3):
    word_len_filter = list(filter(lambda x: x <= i, word_len))
    print(i, len(word_len_filter) / len(word_len))

def read_data(data_path):
    sequence_len = []
    data = pd.read_csv(data_path, sep='\t', header=None, index_col=0,
                       names=['subject', 'subject_name', 'subject_type', 'relation', 'obj', 'tokens', 'labels'])
    
    for _, row in data.iterrows():
        sequence_len.append(len(row['tokens']))
    
    return sequence_len


train_sequence_len = read_data('../data/train.csv')
valid_sequence_len = read_data('../data/valid.csv')
test_sequence_len = read_data('../data/test.csv')
all_sequence_len = []
all_sequence_len.extend(train_sequence_len)
all_sequence_len.extend(valid_sequence_len)
all_sequence_len.extend(test_sequence_len)

print('max train_sequence_len: ', np.max(train_sequence_len))
print('min train_sequence_len: ', np.min(train_sequence_len))
print('avg train_sequence_len: ', np.average(train_sequence_len))
print('median train_sequence_len: ', np.median(train_sequence_len))
print('max valid_sequence_len: ', np.max(valid_sequence_len))
print('min valid_sequence_len: ', np.min(valid_sequence_len))
print('avg valid_sequence_len: ', np.average(valid_sequence_len))
print('median valid_sequence_len: ', np.median(valid_sequence_len))
print('max test_sequence_len: ', np.max(test_sequence_len))
print('min test_sequence_len: ', np.min(test_sequence_len))
print('avg test_sequence_len: ', np.average(test_sequence_len))
print('median test_sequence_len: ', np.median(test_sequence_len))
print('max all_sequence_len: ', np.max(all_sequence_len))
print('min all_sequence_len: ', np.min(all_sequence_len))
print('avg all_sequence_len: ', np.average(all_sequence_len))
print('median all_sequence_len: ', np.median(all_sequence_len))

for i in range(int(np.median(all_sequence_len)), int(np.max(all_sequence_len)), 5):
    seq_len_filter = list(filter(lambda x: x <= i, all_sequence_len))
    print(i, len(seq_len_filter) / len(all_sequence_len))

