# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: create_kg.py

@time: 2018/4/7 19:29

@desc: 1. change entities & relations' names in freebase-FB5M.txt(freebase-FB2M.txt) to be consistent with FB5M.name.txt
       2. create index for entity subject & relation
       3. create map: subject2rel, which will be used for generate negative relation sample in relation network

"""

import os
import sys
from utils import www2fb, pickle_save


def fetch_triple(fb_file, dataset_dir, out_dir):
    basename = os.path.basename(fb_file).split('-')[-1].split('.')[0]

    entities = set()
    relations = set()
    triples = {}
    sub2rel = {}

    with open(fb_file, 'r')as reader:
        for line in reader:
            fields = line.strip().split('\t')

            subject = www2fb(fields[0])
            relation = www2fb(fields[1])
            objects = fields[2].split()

            entities.add(subject)
            relations.add(relation)

            for obj in objects:
                entities.add(www2fb(obj))
                if subject in triples:
                    triples[subject].append((relation, www2fb(obj)))
                else:
                    triples[subject] = [(relation, www2fb(obj))]
                if subject not in sub2rel:
                    sub2rel[subject] = set()
                sub2rel[subject].add(relation)
    
    # agument with triples from dataset
    data_files = [os.path.join(os.path.dirname(dataset_dir), file) for file in os.listdir(dataset_dir)]
    for data_file in data_files:
        with open(data_file, 'r')as reader:
            for line in reader:
                items = line.strip().split('\t')

                subject = www2fb(items[0])
                relation = www2fb(items[1])
                obj = www2fb(items[2])
                
                entities.add(subject)
                entities.add(obj)
                relations.add(relation)
                if subject in triples:
                    triples[subject].append((relation, obj))
                else:
                    triples[subject] = [(relation, obj)]
                if subject not in sub2rel:
                    sub2rel[subject] = set()
                else:
                    sub2rel[subject].add(relation)

    # create index for entity
    idx2entity = list(entities)
    entity2idx = {}
    for idx, entity in enumerate(idx2entity):
        entity2idx[entity] = idx

    # create index for relation
    idx2relation = list(relations)
    relation2idx = {}
    for idx, relation in enumerate(idx2relation):
        relation2idx[relation] = idx

    # create index for subject
    idx2subject = list(triples.keys())
    subject2idx = {}
    for idx, sub in enumerate(idx2subject):
        subject2idx[sub] = idx

    print('num of triples: ', len(triples))
    print('examples of triples: ')
    for key in list(triples.keys())[:10]:
        print('%s : %s' % (key, triples[key]))
    print('num of entities: ', len(entities))
    print('examples of idx2entity: ', idx2entity[:10])
    print('examples of entity2idx: ', list(entity2idx.items())[:10])
    print('num of relations: ', len(relations))
    print('examples of idx2relation: ', idx2relation[:10])
    print('examples of relation2idx: ', list(relation2idx.items())[:10])
    print('num of subjects: ', len(idx2subject))
    print('examples of idx2subject: ', idx2subject[:10])
    print('examples of subject2idx: ', list(subject2idx.items())[:10])
    print('examples of subject2rel: ', list(sub2rel.items())[:10])

    # save as pickle file
    print('store knowledge triples in pickle format...')
    pickle_save(triples, os.path.join(out_dir, basename+'_triple.pkl'))

    print('store idx2entity in pickle format...')
    pickle_save(idx2entity, os.path.join(out_dir, basename+'_idx2entity.pkl'))
    print('store entity2idx in pickle format...')
    pickle_save(entity2idx, os.path.join(out_dir, basename+'_entity2idx.pkl'))

    print('store idx2relation in pickle format...')
    pickle_save(idx2relation, os.path.join(out_dir, basename+'_idx2relation.pkl'))
    print('store relation2idx in pickle format...')
    pickle_save(relation2idx, os.path.join(out_dir, basename+'_relation2idx.pkl'))

    print('store idx2subject in pickle format...')
    pickle_save(idx2subject, os.path.join(out_dir, basename+'_idx2subject.pkl'))
    print('store subject2idx in pickle format...')
    pickle_save(subject2idx, os.path.join(out_dir, basename+'_subject2idx.pkl'))
    
    print('store sub2rel in pickle format...')
    pickle_save(sub2rel, os.path.join(out_dir, basename+'_subject2rel.pkl'))


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'argument error!'

    if not os.path.exists(sys.argv[3]):
        os.makedirs(sys.argv[3])

    fetch_triple(sys.argv[1], sys.argv[2], sys.argv[3])

