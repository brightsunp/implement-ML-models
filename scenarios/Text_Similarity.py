#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/20'

1. Discrete Representation
    One-Hot
    Bag of Words(BOW)
    N-Gram
2. Distributed Representation
    Word2Vec: CBoW / Skip-Gram
'''

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def sim_onehot(text1, text2):
    pass


def sim_bow(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A
    print(tfidf)
    print(sim)
    print('Similarity with BOW:', sim[0, 1])


def sim_ngram(text1, text2):
    pass


if __name__ == '__main__':
    a = np.mat([[0, 1, 2], [0, 1, 2]])
    b = a * a.T
    words1 = '中钢集团 董事长 徐思伟'
    words2 = '中国中钢集团有限公司 董事长 徐思伟'
    sim_bow(words1, words2)

