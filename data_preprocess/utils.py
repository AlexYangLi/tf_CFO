# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: utils.py

@time: 2018/4/7 15:10

@desc:

"""

import pickle


def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    # Manual Correction
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    elif in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    elif in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    elif in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    elif in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    elif in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    elif in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    elif in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    elif in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    elif in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    elif in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    elif in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    elif in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    elif in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    elif in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return in_str


def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri


def pickle_save(obj, save_path):
    with open(save_path, 'wb')as writer:
        pickle.dump(obj, writer)


def pickle_load(load_path):
    with open(load_path, 'rb')as reader:
        return pickle.load(reader)
