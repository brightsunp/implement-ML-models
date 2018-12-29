#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2018/12/29'
'''

import os, math
from Classification_And_Regression_Trees import load_data, evaluate_algrorithm


# hypothesis: sign function
def predict(row, weights):
    activation = weights[0]
    for value, weight in zip(row[:-1], weights[1:]):
        activation += value * weight
    return 1 if activation >= 0.0 else 0


# Stochastic Gradient Descent: estimate Perceptron weights (minimize cost function)
def weights_sgd(train_set, learning_rate, n_epoch):
    '''
    learning_rate: Used to limit the amount each coefficient is corrected each time it is updated.
    n_epoch: The number of times to run through the training data while updating the coefficients.
    '''
    weights = [0.0] * len(train_set[0])
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_set:
            pred = predict(row, weights)
            error = row[-1] - pred
            # calculate square error
            sum_error += math.pow(error, 2)
            # update weights
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] += learning_rate * error * row[i]
        print('>epoch={}, learning_rate={}, error={}'.format(epoch, learning_rate, sum_error))
    return weights


# Perceptron Learning Algorithm
def perceptron(train_set, test_set, learning_rate, n_epoch):
    weights = weights_sgd(train_set, learning_rate, n_epoch)
    predictions = [predict(row, weights) for row in test_set]
    return predictions


if __name__ == '__main__':
    data_dir = './datasets'
    dataset = load_data(os.path.join(data_dir, 'pima-indians-diabetes.csv'))
    learning_rate, n_epoch, n_folds = 0.1, 100, 5
    scores = evaluate_algrorithm(dataset, perceptron, n_folds, learning_rate, n_epoch)
    print('Scores:', scores)
