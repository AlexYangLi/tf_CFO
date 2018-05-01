# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: create_index_type.py

@time: 2018/4/7 20:52

@desc:

"""

import sys
import os
from utils import clean_uri, pickle_load, pickle_save


def get_types_for_entities(types_path, save_dir):
    # entities = pickle_load(entity_path)

    basename = os.path.basename(types_path).split('.')[0]

    entity2type = {}
    entity_types = set()
    with open(types_path, 'r')as reader:
        for index, line in enumerate(reader):
            if index % 100000 == 0:
                print(index)

            fields = line.strip().split('\t')

            entity = clean_uri(fields[0])
            entity_type = clean_uri(fields[2])

            if entity_type != '':
                entity_types.add(entity_type)

                if entity not in entity2type:
                    entity2type[entity] = [entity_type]
                else:
                    entity2type[entity].append(entity_type)
    entity_types = list(entity_types)
    type2idx = {}
    for idx, type in enumerate(entity_types):
        type2idx[type] = idx

    print('num of entity: ', len(entity2type))
    print('examples of entity2type: ', list(entity2type.items())[:10])
    print('num of types: ', len(entity_types))
    print('examples of entity_types: ', entity_types[:10])
 
    print('storing entity2type in pickle format...')
    pickle_save(entity2type, os.path.join(save_dir, basename+'_entity2type.pkl'))
    print('storing entity_types in pickle format...')
    pickle_save(entity_types, os.path.join(save_dir, basename+'_idx2type.pkl'))
    pickle_save(type2idx, os.path.join(save_dir, basename+'_type2idx.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    get_types_for_entities(sys.argv[1], sys.argv[2])

