#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2018/12/29'
'''

import os, math
from Classification_And_Regression_Trees import load_data, evaluate_algrorithm


# rescale dataset columns to range 0-1
def normalize_dataset(dataset):
    min_max_statistics = [(min(values), max(values)) for values in zip(*dataset)]
    for row in dataset:
        for i, value in enumerate(row[:-1]):
            min_value, max_value = min_max_statistics[i]
            # modify row in-place
            row[i] = (value - min_value) / (max_value - min_value)


# hypothesis: sigmoid function
def predict(row, coefficients):
    # first coefficient: intercept, namely bias or b0
    yhat = coefficients[0]
    for value, coefficient in zip(row[:-1], coefficients[1:]):
        yhat += value * coefficient
    return 1.0 / (1.0 + math.exp(-yhat))


# Stochastic Gradient Descent: estimate LR coefficients (minimize cost function)
def coefficients_sgd(train_set, learning_rate, n_epoch):
    '''
    learning_rate: Used to limit the amount each coefficient is corrected each time it is updated.
    n_epoch: The number of times to run through the training data while updating the coefficients.
    '''
    coef = [0.0] * len(train_set[0])
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_set:
            pred = predict(row, coef)
            error = row[-1] - pred
            # calculate square error
            sum_error += math.pow(error, 2)
            # update coefficients
            coef[0] += learning_rate * error * pred * (1 - pred)
            for i in range(len(row) - 1):
                coef[i+1] += learning_rate * error * pred * (1 - pred) * row[i]
        print('>epoch={}, learning_rate={}, error={:.3f}'.format(epoch, learning_rate, sum_error))
    return coef


# Logistic Regression
def logistic_regression(train_set, test_set, learning_rate, n_epoch):
    coef = coefficients_sgd(train_set, learning_rate, n_epoch)
    predictions = [round(predict(row, coef)) for row in test_set]
    return predictions


if __name__ == '__main__':
    data_dir = '../datasets'
    dataset = load_data(os.path.join(data_dir, 'pima-indians-diabetes.csv'))
    normalize_dataset(dataset)
    learning_rate, n_epoch, n_folds = 0.1, 100, 5
    scores = evaluate_algrorithm(dataset, logistic_regression, n_folds, learning_rate, n_epoch)
    print('Scores:', scores)
