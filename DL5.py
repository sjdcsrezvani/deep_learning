# in batch gradient descent (BGD): we through all training samples and calculate cumulative error, now we back propagate
# and adjust weights
# we go through all samples before adjusting weights
# if we have large data it will take more and more time

# stochastic gradient descent (SGD) : randomly pick single data training sample , adjust weights , again randomly pick a
# training sample , again adjust weight , until we get it right
# it is useful when we have big data

# mini batch gradient descent (MBGD) : is like SGD , instead of choosing one randomly picked training sample,
# you will use a batch of randomly picked training samples


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %matplotlib inline

df = pd.read_csv('.\\datasets\\homeprices_banglore.csv')

# now we have to scale our data:

from sklearn import preprocessing

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

scaled_x = sx.fit_transform(df.drop('price', axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))


# now we code our batch gradient descent

def batch_gradient_descent(x, y_true, epochs, learning_rate=0.01):
    number_of_features = x.shape[1]

    w = np.ones(shape=(number_of_features))
    bias = 0
    total_samples = x.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w, x.T) + bias  # T: transpose turning rows into columns  # np.dot(w, x.T) represent :
        # w1 * area + w2 * bedrooms

        w_grad = -(2 / total_samples) * (x.T.dot(y_true - y_predicted))  # derivative for weight and bias
        b_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

        w = w - learning_rate * w_grad
        bias = bias - learning_rate * b_grad

        cost = np.mean(np.square(y_true - y_predicted))  # our cost is mean square error

        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, bias, cost, cost_list, epoch_list


w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_x, scaled_y.reshape(scaled_y.shape[0]), 500)


def predict(area, bedrooms, w, b):
    scaled_x = sx.transform([[area, bedrooms]])[0]
    scaled_price = w[0] * scaled_x[0] + w[1] * scaled_x[1] + b

    price = sy.inverse_transform([[scaled_price]])

    return price


import random


# SGD implement:

def stochastic_gradient_descent(x, y_true, epochs, learning_rate=0.01):
    number_of_features = x.shape[1]

    w = np.ones(shape=(number_of_features))
    bias = 0
    total_samples = x.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):

        random_index = random.randint(0, total_samples - 1)
        sample_x = x[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x.T) + bias

        w_grad = -(2 / total_samples) * (sample_x.T.dot(sample_y - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(sample_y - y_predicted)

        w = w - learning_rate * w_grad
        bias = bias - learning_rate * b_grad

        cost = np.square(sample_y - y_predicted)

        if i % 100 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, bias, cost, cost_list, epoch_list

