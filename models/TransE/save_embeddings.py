# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: save_embeddings.py

@time: 2018/4/16 18:55

@desc:

"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_embedding(model_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    graph = tf.get_default_graph()
    entity_embedding_tensor = graph.get_tensor_by_name('embedding/entity:0')
    relation_embedding_tensor = graph.get_tensor_by_name('embedding/relation:0')

    entity_norm = sess.run(tf.nn.l2_normalize(entity_embedding_tensor, dim=1))
    relation_norm = sess.run(tf.nn.l2_normalize(relation_embedding_tensor, dim=1))
    
    print('shape of entity_embeddings: ', entity_norm.shape)
    print('examples of entity_embddings: ', entity_norm[:2])
    print('shape of relation_embeddings: ', relation_norm.shape)
    print('examples of relation_embeddings: ', relation_norm[:2])

    print('------save entity embedding-----')
    np.save(os.path.join(save_dir, 'entity_transe_embed.npy'), entity_norm)
    print('------save relation embedding----')
    np.save(os.path.join(save_dir, 'relation_transe_embed.npy'), relation_norm)

    print('------save subject embedding-----')
    subject2idx = pickle.load(open('../../data/FB5M_subject2idx.pkl', 'rb'))
    entity2idx = pickle.load(open('../../data/FB5M_entity2idx.pkl', 'rb'))
    subject_embeddings = np.zeros(shape=(len(subject2idx.keys()), entity_norm.shape[1]))
    for sub in subject2idx.keys():
        subject_embeddings[subject2idx[sub]] = entity_norm[entity2idx[sub]]
    np.save(os.path.join(save_dir, 'sub_transe_embed.npy'), subject_embeddings)


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'arguments error!'

    save_embedding(sys.argv[1], sys.argv[2])
