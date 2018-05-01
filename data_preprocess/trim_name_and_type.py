# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: trim_name_and_type.py

@time: 2018/4/8 22:33

@desc: trim entity2name & entity2type to get entity name & entity type only for subject

"""
import os
import sys
from utils import pickle_load, pickle_save


def get_name_type_for_subject(entity2name_path, entity2type_path, triple_path, save_dir):
    entity2name = pickle_load(entity2name_path)
    entity2type = pickle_load(entity2type_path)
    triples = pickle_load(triple_path)

    has_name_count = 0
    has_type_count = 0
    subject2name = {}

    subject2type = {}
    
    for index, subject in enumerate(triples.keys()):
        if subject in entity2name:
            subject2name[subject] = entity2name[subject]
            has_name_count += 1

        if subject in entity2type:
            subject2type[subject] = entity2type[subject]
            has_type_count += 1

    print(has_name_count, len(triples.keys()))
    print(has_type_count, len(triples.keys()))

    pickle_save(subject2name, os.path.join(save_dir, 'trim_subject2name.pkl'))
    pickle_save(subject2type, os.path.join(save_dir, 'trim_subject2type.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 5, 'argument error!'

    if not os.path.exists(sys.argv[4]):
        os.makedirs(sys.argv[4])

    get_name_type_for_subject(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
