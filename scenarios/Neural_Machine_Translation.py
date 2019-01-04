#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/4'
'''

import os, re, string, pickle
import numpy as np
from unicodedata import normalize



# 1. Handle data
def load_data(data_file):
    with open(data_file, 'rt', encoding='utf-8') as f:
        pairs = [line.strip().split('\t')
                 for line in f.readlines()]
        return pairs

def clean_pairs(pairs):
    pass

