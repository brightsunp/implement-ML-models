#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2018/12/26'
'''

import csv, random, math


# 1. Handle data
def load_data(data_file, split_ratio):
    train_set, test_set = [], []
    with open(data_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            row[:4] = [float(s) for s in row[:4]]
            if random.random() < split_ratio:
                train_set.append(row)
            else:
                test_set.append(row)
    return train_set, test_set


# 2. Build model
def calc_distance(instance1, instance2):
    distance = 0
    for ins1, ins2 in zip(instance1, instance2):
        distance += pow((ins1 - ins2), 2)
    return math.sqrt(distance)


def get_neighbors(train_set, instance, k):
    distances = []
    for train_instance in train_set:
        dist = calc_distance(train_instance[:-1], instance[:-1])
        distances.append((train_instance, dist))
    distances.sort(key=lambda item: item[1])
    return [item[0] for item in distances[:k]]


# 3. Predict
def predict(neighbors):
    votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        votes[label] = votes.get(label, 0) + 1
    res = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    return res[0][0]


# 4. Evaluate
def calc_accuracy(test_set, predictions):
    is_correct = [test[-1] == pred for test, pred in zip(test_set, predictions)]
    return sum(is_correct) * 1.0 / len(is_correct)


if __name__ == '__main__':
    data_file = './datasets/iris.data'
    train_set, test_set = load_data(data_file, 0.8)
    print('Train set:', len(train_set))
    print('Test set:', len(test_set))
    predictions = []
    k = 3
    for test_data in test_set:
        neighbors = get_neighbors(train_set, test_data, 3)
        predicion = predict(neighbors)
        predictions.append(predicion)
        print('> predicted={}, actual={}'.format(predicion, test_data[-1]))
    accuracy = calc_accuracy(test_set, predictions)
    print('Accuracy: {:.2%}'.format(accuracy))
