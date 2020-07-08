"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

import pandas as pd
from data_load import DigitDataMNIST
from build_keras_cnn import KerasModel
from datetime import datetime

def csv_saver(predictions, name):

    # Indexes for 28000 test images
    image_index = list(range(1, 28001))

    # Make data frame from predictions and indexes
    df = pd.DataFrame(list(zip(image_index, predictions)), columns=['ImageId', 'Label'], index=None)

    # Name the file the timestamp for differentiation
    if name == 'Time':
        time_now = datetime.now()
        day, hour, minute = str(time_now.day), str(time_now.hour), str(time_now.minute)
        time = day + '-' + hour + '-' + minute
        name = time
    # Define output filename
    output_filename = 'predictions/predictions_{}.csv'.format(name)

    # Save to .csv
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
    model = keras_model.build_and_fit_cnn(predictions=True)
    print('Model history: ', model.history)

    # Make predictions
    pred_labels = model.predict(test_data)
    print('Pred shape: ', pred_labels.shape)
    predictions = []
    for row in pred_labels:
        predictions.append(row.argmax())

    print(predictions)
    print(len(predictions))
    csv_saver(predictions=predictions, name='Time')








