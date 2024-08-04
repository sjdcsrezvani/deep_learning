#  gradient descent is technique that can help you find the parameters in the function  # in supervised learning
#  gradient descent is used during training of the neural network


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

df = pd.read_csv('.\\datasets\\insurance_data.csv')
# we want to scale our data , because affordability is 0 and 1 and age is in (0,100) range

from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(df[['age', 'affordibility']], df.bought_insurance, test_size=0.2,
                                       random_state=25)

x_train_scaled = x_train.copy()  # scaling the data
x_train_scaled['age'] = x_train_scaled['age'] / 100

x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100

model = keras.Sequential([  # building our model neural network
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=5000)

model.evaluate(x_test_scaled, y_test)

# we can see weights and bias in our final function:
coef, intecept = model.get_weights()


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))


def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))


# gradient descent function:

def gradient_descent(age, affordability, y_true, epochs, loss_thresold):
    w1 = w2 = 1
    bias = 0
    learning_rate = 0.5
    n = len(age)

    for i in range(epochs):

        weighted_sum = w1 * age + w2 * affordability + bias
        y_predicted = sigmoid_numpy(weighted_sum)

        loss = log_loss(y_train, y_predicted)

        if loss <= loss_thresold:
            break

        w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
        w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

        bias_d = np.mean(y_predicted - y_true)

        w1 = w1 - learning_rate * w1d
        w2 = w2 - learning_rate * w2d
        bias = bias - learning_rate * bias_d

        print(f'epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

    return w1, w2, bias


gradient_descent(x_train_scaled['age'], x_train_scaled['affordibility'], y_train, 1000, 0.4631)

# create neural network class:

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self, x, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(x.age, x.affordibility, y, epochs, loss_thresold)

    def predict(self, x_test):
        weighted_sum = self.w1 * x_test['age'] + self.w2 * x_test.affordibility + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age, affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        learning_rate = 0.5
        n = len(age)

        for i in range(epochs):

            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)

            loss = log_loss(y_train, y_predicted)

            if loss <= loss_thresold:
                break

            w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)

            w1 = w1 - learning_rate * w1d
            w2 = w2 - learning_rate * w2d
            bias = bias - learning_rate * bias_d
            if i % 50 == 0:
                print(f'epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

        return w1, w2, bias


custom_model = myNN()
custom_model.fit(x_train_scaled, y_train, 500, 0.4631)
