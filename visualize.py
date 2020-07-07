"""
This script visualizes the numbers, not really necessary.
@AugustSemrau
"""

import matplotlib.pyplot as plt
from data_load import DigitDataMNIST


if __name__ == '__main__':

    # Define class for loading
    digit_data = DigitDataMNIST()

    # Load data for building testable model
    X_train_s, X_val, y_train_s, y_val = digit_data.load_digit_mnist(test=False, val_data=True)
    print(X_train_s.shape)
    print(X_val.shape)
    print(y_train_s.shape)
    print(y_val.shape)
    # print(y_train_s)

    # Load data for building final model
    X_train_l, y_train_l = digit_data.load_digit_mnist(test=False, val_data=False)
    print(X_train_l.shape)
    print(y_train_l.shape)

    # Load test data for making predictions
    X_test = digit_data.load_digit_mnist(test=True, val_data=False)
    print(X_test.shape)

    # Use matplotlib for generating an image of a given digit from the data
    X_train_3dim = X_train_s.reshape(X_train_s.shape[0], 28, 28)
    print(X_train_3dim.shape)
    plt.imshow(X_train_3dim[0], cmap='Greys')
    plt.show()
