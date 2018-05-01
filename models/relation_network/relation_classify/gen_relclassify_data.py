import numpy as np
import pandas as pd


def read_data(data_path, word2idx, relation2idx, max_sequence_len=60):
    data_csv = pd.read_csv(data_path, header=None, index_col=None, sep='\t',
                           names=['line_id', 'subject', 'entity_name', 'entity_type', 'relation', 'object',
                                  'tokens', 'labels'])
    # data for training or computing loss
    q_word_ids = []
    sequence_len = []
    pos_rel_ids = []

    for index, row in data_csv.iterrows():
        tokens = row['tokens'].split()
        
        token_idx_has_vector = [word2idx[token] for token in tokens if token in word2idx]
        token_idx_has_vector = token_idx_has_vector[:max_sequence_len]
        pos_rel_idx = relation2idx[row['relation']]     # positive relation's index

        q_word_ids.append(token_idx_has_vector)
        sequence_len.append(len(token_idx_has_vector))
        pos_rel_ids.append(pos_rel_idx)

    # convert to numpy format
    data_size = len(q_word_ids)
    q_word_ids_npy = np.zeros([data_size, max_sequence_len])
    for i in range(data_size):
        q_word_ids_npy[i, :len(q_word_ids[i])] = q_word_ids[i]

    return q_word_ids_npy, np.array(sequence_len), np.array(pos_rel_ids)
