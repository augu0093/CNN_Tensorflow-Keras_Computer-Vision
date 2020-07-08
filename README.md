# Computer-Vision using Tensorflow (Keras API) CNN
By August Semrau Andersen

This project was an entry into the Kaggle competition 'Digit Recognizer'.  
https://www.kaggle.com/c/digit-recognizer/data.  

The goal of the competition is to correctly classify handwritten digits from the MNIST data set using computer vision.

Final classification accuracy achieved on 28000 test MNIST images: 0.99153, pretty decent.

The intent with this project was to learn the basics of computer vision and display proficiency in using Neural Netowrk structures in a practical context.  

### Scripts
The following scripts are used for completing the competition.

1. **data_load.py** loads the MNIST data and converts it from .csv format to array structure.
2. **visualize.py** is a simple reasurance that data is loaded correctly, and can further display the different hand written digits.
3. **build_keras_cnn.py** constructs the CNN and fits it to training data.
4. **optimize_cnn.py** is used for tuning the CNN layer settings for optimal performance.
5. **predictions.py** utilizes the optimized model for printing predictions for entry into the Kaggle competition.



### CNN

Layer structure of the final sequential Keras model was as follows:
1. Convolutional layer; 32 maps and kernel size 5x5
2. Max Pooling layer; size 2x2
3. Convolutional layer; 64 maps and kernel size 5x5
4. Max Pooling layer; size 2x2
5. Flattening layer
6. Dense layer; relu activation with 128 neuron  
7. Dropout layer; dropout-rate = 0.4
8. Dense layer; softmax activation, with 10 neurons (one for each class digits 0-9)

**Compiling** the model is done using Adam optimization, and the loss is crossentropy.

**Training** of the model is performed over 10 epochs.

### TODO
To improve accuracy further, I believe there is some performance to be gained from trying:  

- Using further batch normalization between layers (but need to read up on that).
- Introducing data augmentation for expanded training capabilities.
- Train more epochs and ensemble many models, but this will take along time unless I get CUDA to work haha. 

#### Thank you for reading, I hope you found it interesting

[comment]: <> (Originally the CNN consisted of 6 layers achieving 0.982 acc, but improved with inspiration from an article, 
https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist.)
