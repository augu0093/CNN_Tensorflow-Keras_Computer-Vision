"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

from data_load import DigitDataMNIST




if __name__ == '__main__':
    # Define class for loading
    digit_data = DigitDataMNIST()
    # Load test data
    test_data = DigitDataMNIST.load_digit_mnist(test=True, val_data=False)








