#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/4'

Background:
1989, IBM-PBMT: phrase-based, statistical
2016, Google-NMT: seq2seq, neural
'''

import os, re, string, pickle
import numpy as np
from unicodedata import normalize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint


# 1. Handle Data
def load_data(data_file):
    with open(data_file, 'rt', encoding='utf-8') as f:
        pairs = [line.strip().split('\t')
                 for line in f.readlines()]
        return pairs


# 2. Define Network
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# 3. Compile and Fit Network


# 4. Evaluate Network


# 5. Predict


if __name__ == '__main__':
    data_dir = '../datasets'
    train_set, test_set = load_data(os.path.join(data_dir, 'cmn-eng', 'cmn.txt'))
