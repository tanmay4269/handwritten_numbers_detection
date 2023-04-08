from math import exp
import numpy as np

def sigmoid(x):
    return 1 / (1 + exp(-x))


def activation_function(x):
    # x is a column matrix

    v_func = np.vectorize(sigmoid)
    return v_func(x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def activation_function_prime(x):
    # x is a column matrix

    v_func = np.vectorize(sigmoid_derivative)
    return v_func(x)

