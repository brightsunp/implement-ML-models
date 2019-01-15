#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/3'
'''

import os, math
from common import load_classified_data, evaluate_algrorithm


# rescale dataset columns to range 0-1
def normalize_dataset(dataset):
    min_max_statistics = [(min(values), max(values)) for values in zip(*dataset)]
    for row in dataset:
        for i, value in enumerate(row[:-1]):
            min_value, max_value = min_max_statistics[i]
            # modify row in-place
            row[i] = (value - min_value) / (max_value - min_value)


def predict(row, coefficients):
    # first coefficient: intercept, namely bias or b0
    yhat = coefficients[0]
    for value, coefficient in zip(row[:-1], coefficients[1:]):
        yhat += value * coefficient
    return yhat


# Stochastic Gradient Descent: estimate Linear Regression coefficients
def coefficients_sgd(train_set, learning_rate, n_epoch):
    '''
    learning_rate: Used to limit the amount each coefficient is corrected each time it is updated.
    n_epoch: The number of times to run through the training data while updating the coefficients.
    '''
    coef = [0.0 for _ in range(len(train_set[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_set:
            pred = predict(row, coef)
            error = row[-1] - pred
            # calculate square error
            sum_error += math.pow(error, 2)
            # update coefficients
            coef[0] += learning_rate * error
            for i in range(len(row) - 1):
                coef[i+1] += learning_rate * error * row[i]
        print('>epoch={}, learning_rate={}, error={:.3f}'.format(epoch, learning_rate, sum_error))
    return coef


# Linear Regression
def linear_regression(train_set, test_set, learning_rate, n_epoch):
    coef = coefficients_sgd(train_set, learning_rate, n_epoch)
    predictions = [predict(row, coef) for row in test_set]
    return predictions


# Calculate root mean squared error
def calc_accuracy(test_set, predictions):
    sum_error, n = 0.0, len(test_set)
    for i in range(n):
        error = predictions[i] - test_set[i][-1]
        sum_error += math.pow(error, 2)
    return math.sqrt(sum_error / n)


if __name__ == '__main__':
    data_dir = '../datasets'
    dataset = load_classified_data(os.path.join(data_dir, 'winequality-white.csv'))
    normalize_dataset(dataset)
    learning_rate, n_epoch, n_folds = 0.01, 50, 5
    scores = evaluate_algrorithm(dataset, linear_regression, calc_accuracy, n_folds, learning_rate, n_epoch)
    print('Scores:', scores)
