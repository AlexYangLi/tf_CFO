# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: gen_data.py

@time: 2018/4/17 8:47

@desc:

"""

import random
import numpy as np
import pandas as pd


def read_data(data_path, word2idx, subject2idx, max_sequence_len=60, neg_sample_size=256):
    data_csv = pd.read_csv(data_path, header=None, index_col=None, sep='\t',
                           names=['line_id', 'subject', 'entity_name', 'entity_type', 'relation', 'object',
                                  'tokens', 'labels'])
    # data for training or computing loss: (q, q_len, pos_sub, neg_sub)
    q_word_ids = []
    sequence_len = []
    pos_sub_ids = []
    neg_sub_ids = []

    # data for computing accuracy: (q, q_len, pos_sub, cand_subs)
    q_word_ids_metric = []
    sequence_len_metric = []
    pos_sub_ids_metric = []
    cand_sub_ids_metric = []

    num_subs = len(subject2idx)

    for index, row in data_csv.iterrows():

        tokens = row['tokens'].split()

        token_idx_has_vector = [word2idx[token] for token in tokens if token in word2idx]
        token_idx_has_vector = token_idx_has_vector[:max_sequence_len]
        pos_sub_idx = subject2idx[row['subject']]     # positive relation's index
        
        # negative sampling: randomly sample from subject set
        neg_sub_idx = random.sample(range(num_subs), neg_sample_size)
        if pos_sub_idx in neg_sub_idx:
            neg_sub_idx.remove(pos_sub_idx)
        for neg_sample in neg_sub_idx:
            q_word_ids.append(token_idx_has_vector)
            sequence_len.append(len(token_idx_has_vector))
            pos_sub_ids.append(pos_sub_idx)
            neg_sub_ids.append(neg_sample)

        q_word_ids_metric.append(token_idx_has_vector)
        sequence_len_metric.append(len(token_idx_has_vector))
        pos_sub_ids_metric.append(pos_sub_idx)
        cand_sub_ids_metric.append([pos_sub_idx] + neg_sub_idx)

    # convert to numpy format
    data_size = len(q_word_ids)
    q_word_ids_npy = np.zeros([data_size, max_sequence_len])
    for i in range(data_size):
        q_word_ids_npy[i, :len(q_word_ids[i])] = q_word_ids[i]

    train_data = [q_word_ids_npy, np.array(sequence_len), np.array(pos_sub_ids), np.array(neg_sub_ids)]
    metric_data = [q_word_ids_metric, sequence_len_metric, pos_sub_ids_metric, cand_sub_ids_metric]

    return train_data, metric_data


