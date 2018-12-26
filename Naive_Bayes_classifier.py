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
            row[:-1] = [float(s) for s in row[:-1]]
            row[-1] = int(row[-1])
            if random.random() < split_ratio:
                train_set.append(row)
            else:
                test_set.append(row)
    return train_set, test_set


# 2. Summarize data
def calc_mean(numbers):
    return sum(numbers) * 1.0 / len(numbers)


def calc_stdev(numbers):
    mean = calc_mean(numbers)
    # sample standard deviation
    variance = sum([math.pow(x-mean, 2) for x in numbers]) / (len(numbers)-1)
    return math.sqrt(variance)


def summarize_by_label(data_set):
    sep = {}
    for data in data_set:
        sep.setdefault(data[-1], []).append(data)
    res = {}
    for label, datas in sep.items():
        summaries = [(calc_mean(attr), calc_stdev(attr)) for attr in zip(*datas)]
        summaries.pop()
        res[label] = summaries
    return res


# 3. Calculate probabilities
def calc_prob(x, mean, stdev):
    # Gaussian Probability Density Function
    exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
    pdf_value = (1 / (math.sqrt(2*math.pi)*stdev)) * exponent
    # use log probability
    return math.log(pdf_value)


def calc_label_prob(data_summarized, vector):
    res = {}
    for label, summaries in data_summarized.items():
        for summary, x in zip(summaries, vector):
            mean, stdev = summary
            res[label] = res.get(label, 0) + calc_prob(x, mean, stdev)
            # direct multiplication can lead to floating point underflow
            # (numbers too small to represent in Python)
            # res[label] = res.get(label, 1) * calc_prob(x, mean, stdev)
    return res


# 4. Predict
def predict(probabilities):
    res = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return res[0][0]


# 5. Evaluate
def calc_accuracy(test_set, predictions):
    is_correct = [test[-1] == pred for test, pred in zip(test_set, predictions)]
    return sum(is_correct) * 1.0 / len(is_correct)


if __name__ == '__main__':
    data_file = './datasets/pima-indians-diabetes.csv'
    train_set, test_set = load_data(data_file, 0.8)
    print('Train set:', len(train_set))
    print('Test set:', len(test_set))
    train_summarized = summarize_by_label(train_set)
    predictions = []
    count = 0
    for test_data in test_set:
        probabilities = calc_label_prob(train_summarized, test_data)
        prediction = predict(probabilities)
        predictions.append(prediction)
        if count < 5:
            print('> Log-probilities {} => predicted={}, actual={}'.format(probabilities, prediction, test_data[-1]))
            count += 1
    accuracy = calc_accuracy(test_set, predictions)
    print('Accuracy: {:.2%}'.format(accuracy))
