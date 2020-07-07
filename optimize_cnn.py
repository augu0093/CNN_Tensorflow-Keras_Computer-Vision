"""
This script is used for tuning the CNN.
@AugustSemrau
"""

from build_keras_cnn import KerasModel


if __name__ == '__main__':

    # Initiate CNN model builder
    keras_model = KerasModel()

    # Build and fit CNN to tune-able data set
    model, X_val, y_val = keras_model.build_and_fit_cnn(predictions=False)

    model.evaluate(X_val, y_val)

