# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: gen_relrnn_data.py

@time: 2018/4/11 20:17

@desc: generate data for relation network training

"""

import os
import random
import numpy as np
import pandas as pd
import gc

def read_data(data_path, subject2rel, word2idx, relation2idx, max_sequence_len=60, neg_sample_size=256, restrict=True):
    data_csv = pd.read_csv(data_path, header=None, index_col=None, sep='\t',
                           names=['line_id', 'subject', 'entity_name', 'entity_type', 'relation', 'object',
                                  'tokens', 'labels'])
    # data for training or computing loss: (q, q_len, pos_rel, neg_rel)
    q_word_ids = []
    sequence_len = []
    pos_rel_ids = []
    neg_rel_ids = []

    # data for computing accuracy: (q, q_len, pos_rel, cand_rels)
    q_word_ids_metric = []
    sequence_len_metric = []
    pos_rel_ids_metric = []
    cand_rel_ids_metric = []

    all_relation_idx = set(relation2idx.values())   # all possible relations' indexes

    for index, row in data_csv.iterrows():
        # if index % 1000 == 0:
        #    print(index)

        tokens = row['tokens'].split()

        token_idx_has_vector = [word2idx[token] for token in tokens if token in word2idx]
        token_idx_has_vector = token_idx_has_vector[:max_sequence_len]
        pos_rel_idx = relation2idx[row['relation']]     # positive relation's index
        
        if not restrict:
            # random sampled from relation set
            rest_rel_idx = all_relation_idx - {pos_rel_idx}
            neg_rel_idx = random.sample(rest_rel_idx, neg_sample_size)
        else:
            # step 1: select relations from triples that has the same subject as negative samples,
            neg_rel_idx_with_same_sub = set(map(relation2idx.get, subject2rel[row['subject']])) - {pos_rel_idx}
        
            # step 2: randomly sampled
            if neg_sample_size > len(neg_rel_idx_with_same_sub):
                rest_rel_idx = all_relation_idx - neg_rel_idx_with_same_sub - {pos_rel_idx}
                rest_neg_rel_idx = random.sample(rest_rel_idx, neg_sample_size-len(neg_rel_idx_with_same_sub))
            else:
                rest_neg_rel_idx = []
        
            neg_rel_idx = []
            neg_rel_idx.extend(list(neg_rel_idx_with_same_sub))
            neg_rel_idx.extend(rest_neg_rel_idx)

        for neg_sample in neg_rel_idx:
            q_word_ids.append(token_idx_has_vector)
            sequence_len.append(len(token_idx_has_vector))
            pos_rel_ids.append(pos_rel_idx)
            neg_rel_ids.append(neg_sample)

        q_word_ids_metric.append(token_idx_has_vector)
        sequence_len_metric.append(len(token_idx_has_vector))
        pos_rel_ids_metric.append(pos_rel_idx)
        cand_rel_ids_metric.append([pos_rel_idx] + neg_rel_idx)
       
    # convert to numpy format
    data_size = len(q_word_ids)
    q_word_ids_npy = np.zeros([data_size, max_sequence_len])
    for i in range(data_size):
        q_word_ids_npy[i, :len(q_word_ids[i])] = q_word_ids[i]

    train_data = [q_word_ids_npy, np.array(sequence_len), np.array(pos_rel_ids), np.array(neg_rel_ids)]
    metric_data = [q_word_ids_metric, sequence_len_metric, pos_rel_ids_metric, cand_rel_ids_metric]

    return train_data, metric_data

