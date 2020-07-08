"""
This script builds the CNN for image classification
@AugustSemrau
"""

from data_load import DigitDataMNIST
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#@tf.autograph.experimental.do_not_convert
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


class KerasModel:

    # Init
    def __init__(self):
        # Define class for loading
        digit_data = DigitDataMNIST()
        # Define data for tuning model
        self.X_train, self.X_val, self.y_train, self.y_val = digit_data.load_digit_mnist(test=False, val_data=True)
        # Define data for making final model
        self.X, self.y = digit_data.load_digit_mnist(test=False, val_data=False)

        # Model
        self.model = None

    # Create and fit the Keras CNN
    def build_and_fit_cnn(self, predictions=False):

        # Creating a sequential model
        self.model = Sequential()

        # Adding the layers of the model. 784 - [32C5-P2] - [64C5-P2] - 128 - 10
        self.model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same',  # 32 maps, kernel size 5x5
                              activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=2))  # Pooling size: 2x2
        self.model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same',  # 64 maps, kernel size 5x5
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))  # Pooling size: 2x2
        self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation=tf.nn.relu))  # Dense layer: 128 neurons
        self.model.add(Dropout(0.4))  # Dropout rate: 0.4
        self.model.add(Dense(10, activation=tf.nn.softmax))  # Last dense layer needs 10 neurons, one for each class 0-9

        # Potential layer setup
        # 784 - [32C3-32C3-32C5S2] - [64C3-64C3-64C5S2] - 128 - 10

        # Original layer setup
        # self.model.add(Conv2D(filters=28, kernel_size=3, strides=1, padding='same', activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling size: 2x2
        # self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        # self.model.add(Dense(128, activation=tf.nn.relu))  # Dense layer: 128 neurons
        # self.model.add(Dropout(0.2))  # Dropout rate: 0.2
        # self.model.add(Dense(10, activation=tf.nn.softmax))  # Last dense layer needs 10 neurons

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # print(self.model.summary())

        # Fit the CNN to either split or full training set
        if predictions:
            print('Fitting model:')
            self.model.fit(x=self.X, y=self.y, epochs=10)
            return self.model
        else:
            print('Fitting model:')
            self.model.fit(x=self.X_train, y=self.y_train, epochs=1)
            return self.model, self.X_val, self.y_val
