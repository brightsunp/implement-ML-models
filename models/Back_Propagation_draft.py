#!/usr/bin/python
# coding=utf-8
'''
__author__ = 'sunp'
__date__ = '2019/1/3'

Background:
1958, Rosenblatt-Perceptron: One-layer NN
1969, Minsky <<Perceptron>>: limited to Linear Classification
1986, Rumelhar & Hinton-BP: Two-layer NN
    It can approximate any continuous function infinitely.
1990s, Vapnik-SVM: beat NN
2006, Hinton-Deep Learning: performed up-and-coming in SR & CV
    Big Four: Andrew Ng, Yoshua Bengio, Yann LeCun, Geoffrey Hinton

Learn:
1) forward-propagate an input to calculate an output
2) back-propagate error and train a network
3) apply BP to a real-world predictive modeling problem
'''

import os, math


# Much like Linear Regression
def activate(inputs, weights):
    # first coefficient: intercept, namely bias or b0
    activation = weights[-1]
    for value, coefficient in zip(inputs[:-1], weights[:-1]):
        activation += value * coefficient
    return activation


# Much like Logistic Regression
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(inputs, neuron['weights'])
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
