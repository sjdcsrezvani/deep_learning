import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

tf.config.experimental.list_physical_devices()  # it will list devices that can be used in deep learning
tf.test.is_built_with_cuda()  # if cuda is installed it means that we can use GPU in DL

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # it will load and download the data
# from keras datasets, dataset contains small 60000 images 32x32 which is classified in each row

# typically, we are using CNN for image classification but here we are using ANN (artificial neural network)

x_train.shape  # there's 4 dimension (index,length,width,rgb)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
           'truck']  # saving class names in
# a list by order based on our dataset

# now we scale the images because our model will perform better with scaled data

x_train_scaled = x_train / 255
x_test_scaled = x_test / 255  # now our data is in range(0,1)

# and we change y into categorical values using onehot encoding:
# it is doing same thing as creating dummies in ML
y_train_categorical = keras.utils.to_categorical(
    y_train, num_classes=10, dtype='float32'
)
y_test_categorical = keras.utils.to_categorical(
    y_test, num_classes=10, dtype='float32'
)

# now we create our neural network model:

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(3000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train_categorical, epochs=5)

classes[np.argmax(model.predict(x_test_scaled)[1])]  # it will return maximum index prediction and return the name in


# our classes list

# now performance test:

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# %%timeit -n1 -r1  # this will measure the time it takes for this cell in jupyter notebook
with tf.device('/CPU:0'):  # this model will run on CPU
    cpu_model = get_model()
    cpu_model.fit(x_train_scaled, y_train_categorical, epochs=1)

# %%timeit -n1 -r1
with tf.device('/GPU:0'):  # this model will run and GPU and measure the time
    cpu_model = get_model()
    cpu_model.fit(x_train_scaled, y_train_categorical, epochs=1)

