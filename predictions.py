"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

import pandas as pd
from data_load import DigitDataMNIST
from build_keras_cnn import KerasModel


def csv_saver(predictions, type):

    # Indexes for 28000 test images
    image_index = list(range(1, 28001))

    # Make dataframe from predictions and indexes
    df = pd.DataFrame(list(zip(image_index, predictions)), columns=['ImageId', 'Label'], index=None)
    # predictions_csv = df.to_csv(index=False)
    output_filename = 'predictions/predictions_{}.csv'.format(type)
    return df.to_csv(output_filename, index=False)



if __name__ == '__main__':

    # Define class for loading test data
    digit_data = DigitDataMNIST()
    # Load test data
    test_data = digit_data.load_digit_mnist(test=True, val_data=False)
    print('test_data shape: ', test_data.shape)

    # Initiate CNN model builder
    keras_model = KerasModel()

    # Build and fit CNN to tune-able data set
    model, X_val, y_val = keras_model.build_and_fit_cnn(predictions=False)
    print('Model history: ', model.history)

    # Make predictions
    pred = model.predict(test_data)
    print('Pred shape: ', pred.shape)
    predictions = []
    for row in pred:
        predictions.append(row.argmax())

    print(predictions)
    print(len(predictions))
    csv_saver(predictions=predictions, type="2")
    # print(predictions.argmax())







