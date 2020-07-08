# Computer-Vision using CNN, Tensorflow (Keras API)
By August Semrau Andersen

This project was an entry into the Kaggle competition 'Digit Recognizer'.  
https://www.kaggle.com/c/digit-recognizer/data.  

The goal of the competition is to correctly classify handwritten digits from the MNIST data set using computer vision.

Final classification accuracy on 28000 test MNIST digits: 0.98242.

The intent with this project is to display proficiency in using Neural Netowrk structures in a practical context.  

### Scripts
The following scripts are used for completing the competition.

1. **data_load.py** loads the MNIST data and converts it from .csv format to matrix structure.
2. **visualize.py** is a simple reasurance that data is loaded correctly, and can further display the different hand written digits.
3. **build_keras_cnn.py** constructs the CNN and fits it to training data.
4. **optimize_cnn.py** is used for tuning the CNN layer settings for optimal performance.
5. **predictions.py** utilizes the optimized model for printing predictions for entry into the Kaggle competition.



### CNN

**Layer** structure of the final sequential Keras model was as follows:
1. Convolutional layer, with kernel size 3x3
2. Max Pooling layer, with size 2x2
3. Flattening layer
4. Dense layer, relu activation, with 128 neuron  
5. Dropout layer, with dropout-rate 0.2
6. Dense layer, softmax activation, with 10 neurons (one for each class)

**Compiling** the model is done using Adam optimization, and the loss is crossentropy.

**Training** of the model is performed over 10 epochs.

