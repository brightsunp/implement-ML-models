#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2018/12/29'
'''

import math
from Classification_And_Regression_Trees import *


# sample from dataset: add sample_ratio limit
def sample_dataset(dataset, sample_ratio):
    sample = []
    n_sample = round(len(dataset) * sample_ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        # put back sampling
        sample.append(dataset[index])
    return sample


# sample from features: add n_features limit
def get_best_split(dataset, n_features):
    labels = {row[-1] for row in dataset}
    min_gini, best_index, best_value, best_groups = float('inf'), 0, 0, None
    features = []
    while len(features) < n_features:
        index = random.randrange(len(dataset[0])-1)
        # k different features (optional: random.sample method)
        if index not in features:
            features.append(index)
    for row in dataset:
        for i in features:
            groups = basis_split(i, row[i], dataset)
            gini = calc_gini_index(groups, labels)
            if gini < min_gini:
                min_gini, best_index, best_value, best_groups = gini, i, row[i], groups
    # dict as a tree node {'groups' for build-tree; 'index', 'value' for predict}
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = calc_node_value(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = calc_node_value(left), calc_node_value(right)
        return
    if len(left) <= min_size:
        node['left'] = calc_node_value(left)
    else:
        node['left'] = get_best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = calc_node_value(right)
    else:
        node['right'] = get_best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# build decision trees
def build_tree(train_set, max_depth, min_size, n_features):
    root = get_best_split(train_set, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# predict with a list of bagged trees: voting
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest: add n_trees limit
def random_forest(train_set, test_set, max_depth, min_size, n_trees, sample_ratio, n_features):
    trees = []
    for i in range(n_trees):
        sample = sample_dataset(train_set, sample_ratio)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test_set]
    return predictions


if __name__ == '__main__':
    data_dir = '../datasets'
    dataset = load_data(os.path.join(data_dir, 'banknote_authentication.csv'))
    # default arguments of sklearn.ensemble.RandomForestClassifier:
    # n_estimators=10 or 100, max_depth=None, max_features=sqrt(n_features)
    max_depth, min_size, n_folds = 5, 10, 5
    sample_ratio, n_features = 0.3, int(math.sqrt(len(dataset[0])-1))
    for n_trees in [3, 5, 10]:
        scores = evaluate_algrorithm(dataset, random_forest, n_folds, max_depth, min_size, n_trees, sample_ratio, n_features)
        print('Trees:', n_trees)
        print('Scores:', scores)
        print('Mean Accuracy: {:.3%}'.format(sum(scores) / len(scores)))
