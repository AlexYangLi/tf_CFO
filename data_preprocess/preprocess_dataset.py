# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocess_dataset.py

@time: 2018/4/7 22:11

@desc: pre_process the original simple dataset

"""
import sys
import os
import re
import nltk
from fuzzywuzzy import process, fuzz
from utils import www2fb, pickle_load


def get_indices(src_list, pattern_list):
    indices = None
    for i in range(len(src_list)):
        match = 1
        for j in range(len(pattern_list)):
            if src_list[i+j] != pattern_list[j]:
                match = 0
                break
        if match:
            indices = range(i, i + len(pattern_list))
            break
    return indices


def get_ngram(tokens):
    ngram = []
    for i in range(1, len(tokens)+1):
        for s in range(len(tokens)-i+1):
            ngram.append((" ".join(tokens[s: s+i]), s, i+s))
    return ngram


def reverseLinking(sent, text_candidate):
    tokens = sent.split()
    label = ["O"] * len(tokens)
    text_attention_indices = None
    exact_match = False

    if text_candidate is None or len(text_candidate) == 0:
        return '<UNK>', label, exact_match

    # sorted by length
    for text in sorted(text_candidate, key=lambda x: len(x), reverse=True):
        pattern = r'(^|\s)(%s)($|\s)' % (re.escape(text))
        if re.search(pattern, sent):
            text_attention_indices = get_indices(tokens, text.split())
            break
    if text_attention_indices:
        exact_match = True
        for i in text_attention_indices:
            label[i] = 'I'
    else:
        try:
            v, score = process.extractOne(sent, text_candidate, scorer=fuzz.partial_ratio)
        except :
            print("Extraction Error with FuzzyWuzzy : {} || {}".format(sent, text_candidate))
            return '<UNK>', label, exact_match
        v = v.split()
        n_gram_candidate = get_ngram(tokens)
        n_gram_candidate = sorted(n_gram_candidate, key=lambda x: fuzz.ratio(x[0], v), reverse=True)
        top = n_gram_candidate[0]
        for i in range(top[1], top[2]):
            label[i] = 'I'

    entity_text = []
    for l, t in zip(label, tokens):
        if l == 'I':
            entity_text.append(t)
    entity_text = " ".join(entity_text)
    label = " ".join(label)
    return entity_text, label, exact_match


def augment_dataset(dataset_dir, entity2name_path, entity2type_path, save_dir):
    entity2name = pickle_load(entity2name_path)
    entity2type = pickle_load(entity2type_path)

    data_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)]
    outallfile = open(os.path.join(save_dir, 'all.csv'), 'w')

    name_skipped = 0
    type_skipped = 0
    for data_file in data_files:

        total = 0
        total_exact_match = 0
        fname = os.path.basename(data_file).split('_')[-1].split('.')[0]
        save_path = os.path.join(save_dir, '%s.csv' % fname)
        outfile = open(save_path, 'w')

        print('reading from %s' % os.path.basename(data_file))
        with open(data_file, 'r')as reader:
            for i, line in enumerate(reader):
                if i % 5000 == 0:
                    print(i)

                items = line.strip().split('\t')

                subject = www2fb(items[0])
                relation = www2fb(items[1])
                obj = www2fb(items[2])
                question = ' '.join(nltk.word_tokenize(items[3].lower()))

                lineid = "{}-{}".format(fname, (i + 1))

                if subject not in entity2name.keys():
                    name_skipped += 1
                    print("lineid {} - name not found. {} skipping question.".format(lineid, subject))
                    continue
                
                if subject not in entity2type.keys():
                    type_skipped += 1
                    print("lineid {} - type not found. {} skipping question.".format(lineid, subject))
                    continue

                total += 1

                cand_entity_names = entity2name[subject]
                entity_name, label, exact_match = reverseLinking(question, cand_entity_names)
                if exact_match:
                    total_exact_match += 1

                entity_type = entity2type.get(subject, [])

                line_to_print = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(lineid, subject, entity_name, entity_type, relation, obj,
                                                                    question, label)
                outfile.write(line_to_print + "\n")
                outallfile.write(line_to_print + "\n")

        print("wrote to {}".format(outfile))
        print("Exact Match Entity : {} out of {} : {}".format(total_exact_match, total, total_exact_match / total))
        outfile.close()

    print("wrote to {}".format(outallfile))
    print("name_skipped # questions: {}".format(name_skipped))
    print("type_skipped # questions: {}".format(type_skipped))
    outallfile.close()
    print("DONE!")


if __name__ == '__main__':
    assert len(sys.argv) == 5, 'argument error!'

    if not os.path.exists(sys.argv[4]):
        os.makedirs(sys.argv[4])

    augment_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

