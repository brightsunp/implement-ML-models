#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2018/12/27'
'''

import os
from common import load_classified_data, evaluate_algrorithm


# 1. Handle data
# 2. Create split
def basis_split(index, value, dataset):
    # split the entire dataset by an attribute and an attribute value
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def calc_gini_index(groups, labels):
    total_samples = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        if not group: continue
        group_size = len(group) * 1.0
        score = 0.0
        for label in labels:
            p = [row[-1] for row in group].count(label) / group_size
            score += pow(p, 2)
        # weight the group score by its relative size
        gini += (1.0-score) * (group_size/total_samples)
    return gini


def get_best_split(dataset):
    # Greedy: always select the best split-point for dataset
    labels = {row[-1] for row in dataset}
    min_gini, best_index, best_value, best_groups = float('inf'), 0, 0, None
    for row in dataset:
        for i, value in enumerate(row[:-1]):
            groups = basis_split(i, value, dataset)
            gini = calc_gini_index(groups, labels)
            if gini < min_gini:
                min_gini, best_index, best_value, best_groups = gini, i, value, groups
    # dict as a tree node {'groups' for build-tree; 'index', 'value' for predict}
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


# 3. Build Tree
def calc_node_value(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)


def split(node, max_depth, min_size, depth):
    # each node: haskey('index', 'value', 'left', 'right')
    # leaf node: !haskey('groups')
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
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = calc_node_value(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train_set, max_depth, min_size):
    root = get_best_split(train_set)
    split(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    # preOrder traverse
    if not isinstance(node, dict):
        print('{}label: {}'.format(depth * '  ', node))
        return
    print('{}[X{} < {:.3f}]'.format(depth*'  ', node['index']+1, node['value']))
    print_tree(node['left'], depth+1)
    print_tree(node['right'], depth+1)


# 4. Predict
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train_set, test_set, max_depth, min_size):
    tree = build_tree(train_set, max_depth, min_size)
    print_tree(tree)
    predictions = [predict(tree, row) for row in test_set]
    return predictions


# 5. Evaluate
def calc_accuracy(test_set, predictions):
    is_correct = [test[-1] == pred for test, pred in zip(test_set, predictions)]
    return sum(is_correct) * 1.0 / len(is_correct)


if __name__ == '__main__':
    data_dir = '../datasets'
    dataset = load_classified_data(os.path.join(data_dir, 'banknote_authentication.csv'))
    max_depth, min_size, n_folds = 5, 10, 5
    scores = evaluate_algrorithm(dataset, decision_tree, calc_accuracy, n_folds, max_depth, min_size)
    print('Scores:', scores)
