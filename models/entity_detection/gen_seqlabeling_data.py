# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: gen_seqlabeling_data.py

@time: 2018/4/9 9:40

@desc: generate dataset for sequence labeling task to detect entities appearing in questions

"""
import pickle
import pandas as pd
import numpy as np


def read_data(data_path, word2idx, max_sequence_len=60):

    data_csv = pd.read_csv(data_path, sep='\t', header=None, index_col=None,
                           names=['line_id', 'subject', 'subject_name', 'subject_type', 'relation', 'obj', 'tokens', 'labels'])
    data_size = data_csv.shape[0]

    q_word_ids = np.zeros(shape=[data_size, max_sequence_len])
    q_seq_len = np.zeros(shape=[data_size])
    q_labels = np.zeros(shape=[data_size, max_sequence_len])

    for index, row in data_csv.iterrows():
        tokens = row['tokens'].split()[:max_sequence_len]
        labels = row['labels'].split()[:max_sequence_len]
        
        # make sure there are entity tokens to predict
        if 'I' not in labels:
            continue

        token_has_vector = []
        token_idx_has_vector = []
        label_idx_has_vector = []
        for i in range(len(tokens)):
            if tokens[i] in word2idx:
                token_has_vector.append(tokens[i])
                token_idx_has_vector.append(word2idx[tokens[i]])
                label = 0 if labels[i] == 'O' else 1
                label_idx_has_vector.append(label)

        q_word_ids[index, :len(token_idx_has_vector)] = token_idx_has_vector
        q_labels[index, :len(label_idx_has_vector)] = label_idx_has_vector
        q_seq_len[index] = len(token_idx_has_vector)

    return q_word_ids, q_seq_len, q_labels


def read_data_for_test(data_path, word2idx, max_sequence_len=60):
    data_csv = pd.read_csv(data_path, sep='\t', header=None, index_col=None,
                           names=['line_id', 'subject', 'subject_name', 'subject_type', 'relation', 'obj', 'tokens',
                                  'labels'])
    data_size = data_csv.shape[0]

    q_lineid = []
    q_word_ids = np.zeros(shape=[data_size, max_sequence_len])
    q_seq_len = np.zeros(shape=[data_size])
    q_labels = np.zeros(shape=[data_size, max_sequence_len])

    for index, row in data_csv.iterrows():
        tokens = row['tokens'].split()[:max_sequence_len]
        labels = row['labels'].split()[:max_sequence_len]

        # make sure there are entity tokens to predict
        # if 'I' not in labels:
        #     continue
        q_lineid.append(row['line_id'])

        token_has_vector = []
        token_idx_has_vector = []
        label_idx_has_vector = []
        for i in range(len(tokens)):
            if tokens[i] in word2idx:
                token_has_vector.append(tokens[i])
                token_idx_has_vector.append(word2idx[tokens[i]])
                label = 0 if labels[i] == 'O' else 1
                label_idx_has_vector.append(label)

        q_word_ids[index, :len(token_idx_has_vector)] = token_idx_has_vector
        q_labels[index, :len(label_idx_has_vector)] = label_idx_has_vector
        q_seq_len[index] = len(token_idx_has_vector)

    return q_lineid, q_word_ids, q_seq_len, q_labels

