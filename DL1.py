import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path)  # loading hand writen keras library

# we have to flatten our 2 dimension array into 1 dimension array for our neural network

x_train_flatten = x_train.reshape(len(x_train), 28 * 28)  # it will flatten and will keep number of samples and the data
x_train_flatten.shape
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)

# now we create simple neural network with just 2 layers : input layer and output layer

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
    # we are creating our neural network , 10 is number
    # of neurons in second layer and our number of neurons in input layer is 784 , and we are using sigmoid function
    # as activation
])

model.compile(optimizer='adam',  # we are using adam to reach global optimum
              loss='sparse_categorical_crossentropy',  # defining that our output is categorical and sparse means it is
              # digits
              metrics=['accuracy']  # and we want accuracy in our model
              )

model.fit(x_train_flatten, y_train, epochs=5)  # epoch means number of iterations

# we can scale the values to increase our accuracy
x_train = x_train / 255
x_test = x_test / 255  # now values are between 0 and 1

model.evaluate(x_test_flatten, y_test)  # this is the score and evaluate our accuracy on test dataset

y_predicted = model.predict(x_test_flatten)  # it will return scores for each output neuron
np.argmax(y_predicted[3])  # it will return index of maximum argument

y_predicted_labels = [np.argmax(i) for i in y_predicted]  # there is list for each prediction ,so we classify it by
# this function

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)  # tensorflow has confusion matrix function

# now we can add new hidden layer to improve our model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # this layer will flatten our array
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=5)


# creating tensorboard by callbacks:

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)  # creating object for our tensorboard
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=5, callbacks=[tb_callback])  # putting our data in that object
