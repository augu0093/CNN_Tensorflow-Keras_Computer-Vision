"""
This script loads the MNIST numbers data set for utilization.
@AugustSemrau
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

def load_digit_mnist():

    df_train = pd.read_csv(train_path)


class DigitDataMNIST:

    # Initialize
    def __init__(self):
        self.train_path = './data/train.csv'
        self.test_path = './data/test.csv'
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.test_data = None
        # self.transform = transform

    # Load the data, either for training the model, validation trained performance or producing Kaggle submission
    def load_digit_mnist(self, test=False, val_data=False):
        # Load training data and split label from data
        train_data = pd.read_csv(self.train_path)
        self.X_train = np.asarray(train_data.iloc[:, 1:]).reshape(-1, 28, 28)  # .astype(float);
        self.y_train = np.asarray(train_data.iloc[:, 0])

        # Load test data
        test_data = pd.read_csv(self.test_path)
        self.test_data = np.asarray(test_data.iloc[:, 0:]).reshape(-1, 28, 28)

        # Return test data for Kaggle submission
        if test:
            return self.test_data

        # Return training data
        if val_data:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.2, random_state=0)
            return self.X_train, self.X_val, self.y_train, self.y_val
        else:
            return self.X_train, self.y_train

    # Get shape of the data
    def __XtShape__(self):
        return self.X_train.shape

    def __XvShape__(self):
        return self.X_val.shape

    def __YtShape__(self):
        return self.y_train.shape

    def __YvShape__(self):
        return self.y_val.shape

    def __TestShape__(self):
        return self.test_data.shape

    # def __getitem__(self, idx):
    #     item = self.X[idx]
    #     label = self.Y[idx]
    #
    #     if self.transform:
    #         item = self.transform(item)
    #
    #     return (item, label)