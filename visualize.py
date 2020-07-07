"""
This script visualizes the numbers, not really necessary.
@AugustSemrau
"""

import matplotlib.pyplot as plt
from data_load import DigitDataMNIST



if __name__ == '__main__':

    # Define class for loading
    digit_data = DigitDataMNIST()

    # Load data
    X_train, X_val, y_train, y_val = digit_data.load_digit_mnist(test=False, val_data=True)
    print(X_train[0])

    # Use matplotlib for generating image of digit
    plt.imshow(X_train[0], cmap='Greys')
    plt.show()



