#  use log loss or binary cross entropy not mean squared error

import numpy as np

y_predicted = np.array([1, 1, 0, 0, 1])
y_true = np.array([0.30, 0.7, 1, 0, 0, 0.5])


def mae(y_true, y_predicted):  # the function for mean absolute error
    total_error = 0
    for yt, yp in zip(y_true, y_predicted):
        total_error += abs(yt - yp)
    print('total error:', total_error)
    mae = total_error / len(y_true)
    return mae


np.mean(np.abs(y_predicted - y_true))  # mean absolute function using pandas in 1 line

# log loss function
epsilon = 1e-15
y_predicted_new = [max(i, epsilon) for i in y_predicted]  # we are changing values of 0 and 1 with epsilon and
# 1-epsilon
y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]  # because log(0) will be infinite, and it is not
# acceptable
y_predicted_new = np.array(y_predicted_new)

log_loss = -np.mean(
    y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))  # log loss function using


# pandas

def log_loss(y_true, y_predicted):  # everything put in one function
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))


