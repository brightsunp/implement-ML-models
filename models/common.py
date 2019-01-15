#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/15'
'''

import csv, random


def load_classified_data(data_file):
    # features float & labels int
    dataset = []
    with open(data_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            try:
                row[:-1] = [float(s) for s in row[:-1]]
                row[-1] = int(row[-1])
                dataset.append(row)
            except ValueError:
                continue
    return dataset


def evaluate_algrorithm(dataset, algorithm, calc_accuracy, n_folds, *args):
    accuracies = []
    folds = _cross_validation_split(dataset, n_folds)
    for fold in folds:
        train_set, test_set = list(folds), list(fold)
        train_set.remove(fold)
        # convert list-of-lists to list
        train_set = sum(train_set, [])
        predictions = algorithm(train_set, test_set, *args)
        accuracy = calc_accuracy(test_set, predictions)
        accuracies.append(accuracy)
    return accuracies


def _cross_validation_split(dataset, n_folds):
    dataset_split = []
    # copy.deepcopy
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
