"""
This script builds the CNN for image classification
@AugustSemrau
"""

from data_load import DigitDataMNIST
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

        # Adding the layers of the model
        self.model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))  # Kernel size: 3x3
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling size: 2x2
        self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation=tf.nn.relu))  # Dense layer: 128 neurons
        self.model.add(Dropout(0.2))  # Dropout rate: 0.2
        self.model.add(Dense(10, activation=tf.nn.softmax))  # Last dense layer needs 10 neurons, one for each class 0-9

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Fit the CNN to either split or full training set
        if predictions:
            return self.model.fit(x=self.X, y=self.y, epochs=10)
        else:
            return self.model.fit(x=self.X_train, y=self.y_train, epochs=10), self.X_val, self.y_val
