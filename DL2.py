# activation functions

# sigmoid function acts like logit or logistic regression and when we have multiclass classification it comes handy
# and the outcome probability is between 0 and 1

# tanh function is similar to sigmoid but instead of giving answer between 0 and 1 it gives the answer between -1 and
# 1 general guideline : use sigmoid in output layer, all other places try to use tanh issues with sigmoid and tanh in
# higher incomes the output won't change that much look at log (nemudar) and our slope or derivative will be 0 and
# learning will be slow  ( vanishing gradients ) problem

# ReLU function: if income is less than 0 then outcome is 0 , for values greater than 0 outcome will be y=x
# for hidden layers if you are not sure which activation function to use just use ReLU as your default choice
# Relu won't change the outcome and its light function, and it will be used for hidden layers

# leaky ReLU function: for values less than 0 , y= 0.1x and for values greater than 0, y= x

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return max(0.1 * x, x)
