# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: gen_data.py

@time: 2018/4/18 8:58

@desc:

"""

import time
import random
import numpy as np
import pandas as pd


def get_typevec(types, type2idx):
    # to save memory, just store the index which should be 1
    if types:
        return [type2idx[typ] for typ in types]
    else:
        return []


def read_data(data_path, word2idx, idx2subject, subject2idx, subject2type, type2idx, max_sequence_len=60, neg_sample_size=256):
    data_csv = pd.read_csv(data_path, header=None, index_col=None, sep='\t',
                           names=['line_id', 'subject', 'entity_name', 'entity_type', 'relation', 'object',
                                  'tokens', 'labels'])

    q_word_ids = []
    sequence_len = []
    pos_typevecs = []
    cand_subs = []
    cand_typevecs = []

    num_type = len(type2idx)
    num_sub = len(idx2subject)

    for index, row in data_csv.iterrows():
        #if index % 1000 == 0:
        #    print(index)
        tokens = row['tokens'].split()
        token_idx_has_vector = [word2idx[token] for token in tokens if token in word2idx]
        token_idx_has_vector = token_idx_has_vector[:max_sequence_len]

        pos_sub = row['subject']
        if pos_sub not in subject2type:
            continue

        pos_sub_idx = subject2idx[pos_sub]
        pos_typevec = get_typevec(subject2type[pos_sub], type2idx)     # postive subject type vector
        
        neg_sub_idx = random.sample(range(num_sub), neg_sample_size)    # sampling from sub_index is more efficent than sampling from subject(string)
        
        cand_sub_idx = [pos_sub_idx]    # make sure index 0 is positive subject
        cand_sub_idx.extend(neg_sub_idx)

        can_typevec = [get_typevec(subject2type.get(idx2subject[sub_idx], None), type2idx) for sub_idx in cand_sub_idx]
        
        q_word_ids.append(token_idx_has_vector)
        sequence_len.append(len(token_idx_has_vector))
        pos_typevecs.append(pos_typevec)
        cand_subs.append(cand_sub_idx)
        cand_typevecs.append(can_typevec)

    # convert to numpy format
    data_size = len(q_word_ids)
    q_word_ids_npy = np.zeros([data_size, max_sequence_len])
    for i in range(data_size):
        q_word_ids_npy[i, :len(q_word_ids[i])] = q_word_ids[i]
    print('done')
    return q_word_ids_npy, np.array(sequence_len), pos_typevecs, cand_subs, cand_typevecs

