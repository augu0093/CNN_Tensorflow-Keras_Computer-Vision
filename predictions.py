"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

from data_load import DigitDataMNIST
from build_keras_cnn import KerasModel



if __name__ == '__main__':
    # Define class for loading
    digit_data = DigitDataMNIST()
    # Load test data
    test_data = DigitDataMNIST.load_digit_mnist(test=True, val_data=False)

    # Initiate CNN model builder
    keras_model = KerasModel()

    # Build and fit CNN to tune-able data set
    model, X_val, y_val = keras_model.build_and_fit_cnn(predictions=False)
    # model.save('model_test_1')
    print('Model history: ', model.history)
    print('X_val shape: ', X_val.shape)
    print('y_val shape: ', y_val.shape)
    print('Model evaluation: ', model.evaluate(X_val, y_val))






