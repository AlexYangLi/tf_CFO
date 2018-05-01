# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: inference.py

@time: 2018/4/22 14:32

@desc:

"""

import os
import random
from argparse import ArgumentParser
import numpy as np
import pickle
import pandas as pd
from models import EntDect, RelNet, SubTransE, SubTypeVec

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def pickle_load(data_path):
    return pickle.load(open(data_path, 'rb'))


def pickle_save(obj, data_path):
    pickle.dump(obj, open(data_path, 'wb'))


def get_all_ngram(text):
    tokens = text.split()
    all_ngrams = {}     # only consider unigram, bigram, trgram
    for i in range(1, 4):
        all_ngrams[i] = find_ngrams(tokens, i)

    return all_ngrams


def find_ngrams(tokens, n):
    ngrams = list(set(zip(*[tokens[i:] for i in range(n)])))
    ngrams = [' '.join(ngram) for ngram in ngrams]
    return ngrams


def read_data(data_path, word2idx, relation2idx, subject2idx, max_sequence_len=60):
    data_csv = pd.read_csv(data_path, header=None, index_col=None, sep='\t',
                           names=['line_id', 'subject', 'entity_name', 'entity_type', 'relation', 'object',
                                  'tokens', 'labels'])

    data_size = data_csv.shape[0]
    q_lineid = []
    questions = []
    q_word_ids = np.zeros(shape=(data_size, max_sequence_len))
    q_seq_len = np.zeros(shape=data_size)
    gold_sub_ids = []
    gold_rel_ids = []

    for index, row in data_csv.iterrows():
        tokens = row['tokens'].split()

        token_has_vector = [token for token in tokens if token in word2idx]
        token_has_vector = token_has_vector[:max_sequence_len]
        token_idx_has_vector = [word2idx[token] for token in token_has_vector]

        q_lineid.append(row['line_id'])
        questions.append(' '.join(token_has_vector))
        q_word_ids[index, :len(token_idx_has_vector)] = token_idx_has_vector
        q_seq_len[index] = len(token_idx_has_vector)
        gold_sub_ids.append(subject2idx[row['subject']])
        gold_rel_ids.append(relation2idx[row['relation']])

    return q_lineid, questions, q_word_ids, q_seq_len, gold_sub_ids, gold_rel_ids


def link_entity(mentions, name2subject, ngram2subject, kb_triple, subject2idx, relation2idx):
    # the suffix "ids" means we store the index of subject(relation) rather than subject(relation) itself
    cand_sub_ids = []
    cand_rel_ids = []
    cand_subrel_ids = []

    match_count = [0, 0, 0, 0]  # index 0 for exact match, index 1, 2, 3 for unigram / bigram / trigram match
    for mention in mentions:
        cand_sub = set()
        cand_rel = set()
        cand_subrel = []

        if mention in name2subject:
            match_count[0] += 1
            cand_sub.update(name2subject[mention])
        else:
            ngrams = get_all_ngram(mention)

            for i in [3, 2, 1]:
                for ngram in ngrams[i]:
                    if ngram in ngram2subject:
                        cand_sub.update(list(zip(*ngram2subject[ngram]))[0][:256])

                if len(cand_sub) > 0:
                    match_count[i] += 1
                    break
        #if len(cand_sub) > 256:
        #    cand_sub = random.sample(cand_sub, 256)
        for sub in list(cand_sub):
            rels = set([rel for rel, _ in kb_triple[sub]])
            cand_rel.update(rels)
            cand_subrel.extend([(sub, rel) for rel in list(rels)])

        cand_sub_ids.append([subject2idx[sub] for sub in list(cand_sub)])
        cand_rel_ids.append([relation2dix[rel] for rel in list(cand_rel)])
        cand_subrel_ids.append([(subject2idx[sub], relation2dix[rel]) for sub, rel in cand_subrel])
        
    print('exact match: {} / {} '.format(match_count[0], len(mentions)))
    print('trigram match: {} / {}'.format(match_count[3], len(mentions)))
    print('bigram match: {} / {}'.format(match_count[2], len(mentions)))
    print('unigram match: {} / {}'.format(match_count[1], len(mentions)))

    return cand_sub_ids, cand_rel_ids, cand_subrel_ids


def inference(gold_sub_ids, gold_rel_ids, cand_subrel_ids, rel_scores, sub_scores):
    for alpha in [0.45]:
        subrel_hit = 0
        sub_hit = 0
        rel_hit = 0

        data_size = len(gold_sub_ids)
        for i in range(data_size):
            subrel_scores = {}
            for (sub_id, rel_id) in cand_subrel_ids[i]:
                score = rel_scores[i][rel_id] * (alpha + (1 - alpha)*sub_scores[i][sub_id])
                subrel_scores[(sub_id, rel_id)] = score

            subrel_scores = sorted(subrel_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(subrel_scores) == 0:
                continue
           
            top_sub_id = subrel_scores[0][0][0]
            top_rel_id = subrel_scores[0][0][1]
            if top_sub_id == gold_sub_ids[i]:
                sub_hit += 1
            if top_rel_id == gold_rel_ids[i]:
                rel_hit += 1
            if top_sub_id == gold_sub_ids[i] and top_rel_id == gold_rel_ids[i]:
                subrel_hit += 1

        print('alpha: %f, sub acc: %f, rel acc: %f, (sub, rel): %f' % (alpha, sub_hit / data_size, rel_hit / data_size,
                                                                        subrel_hit / data_size))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='../../data/test.csv', help='path to test data')
    parser.add_argument('--word2idx', type=str, default='../../data/fb_word2idx.pkl', help='path to word2idx.pkl')
    parser.add_argument('--rel2idx', type=str, default='../../data/FB5M_relation2idx.pkl',
                        help='path to relation2idx.pkl')
    parser.add_argument('--sub2idx', type=str, default='../../data/FB5M_subject2idx.pkl',
                        help='path to subject2idx.pkl')
    parser.add_argument('--idx2sub', type=str, default='../../data/FB5M_idx2subject.pkl',
                        help='path to idx2subject.pkl')
    parser.add_argument('--sub2type', type=str, default='../../data/trim_subject2type.pkl',
                        help='path to subject2type')
    parser.add_argument('--type2idx', type=str, default='../../data/FB5M_type2idx.pkl', help='path to type2idx.pkl')
    parser.add_argument('--name2sub', type=str, default='../../data/name2subject.pkl',
                        help='path to subject2name.pkl')
    parser.add_argument('--ngram2sub', type=str, default='../../data/ngram2subject.pkl',
                        help='path to subngram2entity.pkl')
    parser.add_argument('--kb', type=str, default='../../data/FB5M_triple.pkl', help='path to knowledge graph')
    parser.add_argument('--entdect_type', type=str, required=True,
                        help='model type of entity detection, options are [lstm | lstm_crf | bilstm | bilstm_crf]')
    parser.add_argument('--subnet_type', type=str, required=True,
                        help='model type of subject network, options are [transe | typevec]')
    parser.add_argument('--entdect', type=str, required=True, help='path to entity detection tensorflow model')
    parser.add_argument('--relnet', type=str, required=True, help='path to relation network tensorflow model')
    parser.add_argument('--subnet', type=str, required=True, help='path to subject network tensorflow model')
    args = parser.parse_args()

    # load needed data
    print('loading word2idx...')
    word2idx = pickle_load(args.word2idx)
    print('loading relation2idx...')
    relation2dix = pickle_load(args.rel2idx)
    print('loading idx2subject...')
    idx2subject = pickle_load(args.idx2sub)
    print('loading subject2idx...')
    subject2idx = pickle_load(args.sub2idx)
    print('loading type2idx...')
    type2idx = pickle_load(args.type2idx)
    print('loading subject2type...')
    subject2type = pickle_load(args.sub2type)
    print('loading name2subject')
    name2subject = pickle_load(args.name2sub)
    print('loading ngram2subject...')
    ngram2subject = pickle_load(args.ngram2sub)
    print('loading knowledge graph...')
    kb_triple = pickle_load(args.kb)

    # load model
    print('load entity detection model...')
    entdect = EntDect(args.entdect_type, args.entdect)
    print('load relation network model...')
    relnet = RelNet(args.relnet)
    if args.subnet_type == 'typevec':
        subnet = SubTypeVec(args.subnet)
    else:
        subnet = SubTransE(args.subnet)

    # load test data
    print('loading test data...')
    q_lineid, questions, q_word_ids, q_seq_len, gold_sub_ids, gold_rel_ids = read_data(args.data, word2idx,
                                                                                       relation2dix, subject2idx)

    # '''step1: entity detection: find possible subject mention in question'''
    mentions = entdect.infer((questions, q_word_ids, q_seq_len))
    
    # '''step2: entity linking: find possible subjects responding to subject mention;
    #           search space reduction: generate candidate (subject, relation) pair according to possible subjects
    # '''
    cand_sub_ids, cand_rel_ids, cand_subrel_ids = link_entity(mentions, name2subject, ngram2subject,
                                                              kb_triple, subject2idx, relation2dix)
    
    # '''step3: relation scoring: compute score for each candidate relations'''
    rel_scores = relnet.infer((q_word_ids, q_seq_len, cand_rel_ids))
    
    # '''step4: subject scoring: compute score for each candidate subjects'''
    if args.subnet_type == 'typevec':
        cand_sub_typevecs = []
        for can_sub in cand_sub_ids:
            type_vecs = []
            for sub_id in can_sub:
                types = subject2type.get(idx2subject[sub_id], [])
                type_ids = [type2idx[type] for type in types]
                type_vecs.append(type_ids)
            cand_sub_typevecs.append(type_vecs)
        sub_scores = subnet.infer((q_word_ids, q_seq_len, cand_sub_ids, cand_sub_typevecs))
    else:
        sub_scores = subnet.infer((q_word_ids, q_seq_len, cand_sub_ids))
  
    # '''step5: inference'''
    inference(gold_sub_ids, gold_rel_ids, cand_subrel_ids, rel_scores, sub_scores)


