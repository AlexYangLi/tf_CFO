# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: create_ngram2subject.py

@time: 2018/4/22 0:15

@desc: create dict: key: subject names' ngrams, value: list of (subject_name, subject_id)
       create dict: key: subject name, value: list of subject_id

"""

import sys
import os
from utils import pickle_load, pickle_save


def get_all_ngrams(tokens):
    all_ngrams = set()
    max_n = min(len(tokens), 3)
    for n in range(1, max_n+1):
        ngrams = find_ngrams(tokens, n)
        all_ngrams = all_ngrams | ngrams
    return all_ngrams


def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)


def get_name_ngrams(subject_name):
    name_tokens = subject_name.split()
    name_ngrams = get_all_ngrams(name_tokens)
    return name_ngrams


def create_ngram2subject(subject2name, save_dir):
    ngram2subject = {}
    name2subject = {}
    for subject_id, subject_names in subject2name.items():
        for subject_name in subject_names:
            if subject_name not in name2subject:
                name2subject[subject_name] = [subject_id]
            else:
                name2subject[subject_name].append(subject_id)

            name_ngrams = get_name_ngrams(subject_name)

            for ngram_tuple in name_ngrams:
                ngram = ' '.join(ngram_tuple)
                if ngram in ngram2subject.keys():
                    ngram2subject[ngram].append((subject_id, subject_name))
                else:
                    ngram2subject[ngram] = [(subject_id, subject_name)]

    print('num of subject names: ', len(name2subject))
    print('examples of name2subject: ', list(name2subject.items())[:10])
    print('num of ngram: ', len(ngram2subject))
    print('examples of ngram2subject: ', list(ngram2subject.items())[:10])

    print('save ngram2subject in pickle format...')
    pickle_save(ngram2subject, os.path.join(save_dir, 'ngram2subject.pkl'))
    print('save name2subject in pickle format...')
    pickle_save(name2subject, os.path.join(save_dir, 'name2subject.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'
    
    subject2name = pickle_load(sys.argv[1])
    
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    create_ngram2subject(subject2name, sys.argv[2])

