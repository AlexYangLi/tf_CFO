import os
import pickle
import numpy as np
import random


class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.n_entity = 0
        self.relation_dict = {}
        self.n_relation = 0
        self.golden_triple_pool = {}    # read directly from file, key:h, value: list of (r, t)
        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()

    def load_dicts(self):
        entity_dict_file = 'FB5M_entity2idx.pkl'
        print('-----Loading entity dict-----')
        self.entity_dict = pickle.load(open(os.path.join(self.data_dir, entity_dict_file), 'rb'))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))

        relation_dict_file = 'FB5M_relation2idx.pkl'
        print('-----Loading relation dict-----')
        self.relation_dict = pickle.load(open(os.path.join(self.data_dir, relation_dict_file), 'rb'))
        self.n_relation = len(self.relation_dict.values())
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        kb_file = 'FB5M_triple.pkl'
        print('-----Loading knowledge graph-----')
        self.golden_triple_pool = pickle.load(open(os.path.join(self.data_dir, kb_file), 'rb'))

        print('-----Loading training triples-----')
        # all triples are considered as training triples because we want to make sure all the entities
        # and relations' embeddings can be trained, which we need for further study.
        for h in self.golden_triple_pool.keys():
            triples_for_h = [(self.entity_dict[h], self.entity_dict[t], self.relation_dict[r])
                             for r, t in self.golden_triple_pool[h]]
            self.training_triples.extend(triples_for_h)
        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))

        # validation triples and test triples are just sampled from training triples
        print('-----Loading validation triples-----')
        self.validation_triples = random.sample(self.training_triples, 1000)
        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))

        print('-----Loading test triples------')
        self.test_triples = random.sample(self.training_triples, 1000)
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))

    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple:
            end = min(start + batch_size, self.n_training_triple)
            yield [self.training_triples[i] for i in rand_idx[start:end]]
            start = end

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.choice(self.entities)
                        else:
                            tail_neg = random.choice(self.entities)
                        if head_neg not in self.golden_triple_pool or \
                                (tail_neg, relation) not in self.golden_triple_pool[head_neg]:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg))
