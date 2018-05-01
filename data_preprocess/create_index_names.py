# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: create_index_names.py

@time: 2018/4/7 20:15

@desc: get names or alias for each entity in knowledge triples

"""

import sys
import os
import nltk
from utils import clean_uri, pickle_load, pickle_save


def get_names_for_entities(names_path, save_dir):
    # entities = pickle_load(entity_path)
    
    basename = os.path.basename(names_path).split('.')[0]

    entity2name = {}
    name2entity = {}
    with open(names_path, 'r')as reader:
        for index, line in enumerate(reader):
            if index % 100000 == 0:
                print(index)

            fields = line.strip().split('\t')

            entity = clean_uri(fields[0])
            entity_name = ' '.join(nltk.word_tokenize(clean_uri(fields[2].lower())))
 
            if entity not in entity2name:
                entity2name[entity] = [entity_name]
            else:
                entity2name[entity].append(entity_name)
                
            if entity_name not in name2entity:
               name2entity[entity_name] = [entity]
            else:
               name2entity[entity_name].append(entity)
    
    print('num of entity: ', len(entity2name))
    print('examples of entity2name: ', list(entity2name.items())[:10])
    print('num of entityname: ', len(name2entity))
    print('examples of name2entity: ', list(name2entity.items())[:10])

    print('storing entity2name in pickle format...')
    pickle_save(entity2name, os.path.join(save_dir, basename+'_entity2name.pkl'))

    print('storing name2entity in pickle format...')
    pickle_save(name2entity, os.path.join(save_dir, basename+'_name2entity.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'arguments error!'

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    get_names_for_entities(sys.argv[1], sys.argv[2])

