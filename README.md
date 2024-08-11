# Automated-Face-Mask-Detection-Using-CNNs
# Project Overview
This project involves building a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask in an image. The model is trained using a dataset from Kaggle containing images of people with and without masks.
# Dataset
- Source: Kaggle
- Content: The dataset consists of 7,553 images in total, with 3,725 images of people wearing masks and 3,828 images of people without masks.
- Size: The images are resized to 128x128 pixels and converted to RGB format for uniformity and to facilitate model training.
# Data Preprocessing
1. Loading and Resizing: All images are loaded and resized to 128x128 pixels.
2. Labeling
   - Images of people with masks are labeled as 1.
   - Images of people without masks are labeled as 0.
3. Normalization: The pixel values are scaled to the range [0, 1] by dividing by 255.
4. Train-Test Split: The data is split into training and testing sets with an 80-20 split.
# Model Architecture
The CNN model is built using the Keras library with the following layers:
1. Convolutional Layers: Two Conv2D layers with 64 and 32 filters respectively, each followed by a MaxPooling2D layer.
2. Flatten Layer: Converts the 2D matrix data to a 1D vector.
3. Dense Layers
   - A fully connected layer with 128 units and ReLU activation.
   - A dropout layer to prevent overfitting.
   - A fully connected layer with 64 units and ReLU activation.
   - Another dropout layer.
4. Output Layer: A Dense layer with 2 units and softmax activation for binary classification.
# Training
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 10
The model is trained with an 80-20 validation split on the training data.
# Evaluation
The trained model is used to predict whether a person in a given image is wearing a mask. The image is preprocessed similarly to the training data and then fed into the model to obtain predictions.
# Prediction
The trained model is used to predict whether a person in a given image is wearing a mask. The image is preprocessed similarly to the training data and then fed into the model to obtain predictions.
# Results
The model achieved high accuracy on both the training and validation datasets, indicating its effectiveness in distinguishing between images of people with and without masks.
# Future Work
- Improve model performance with more diverse datasets.
- Integrate real-time mask detection using video feed.
- Enhance the model to handle different lighting conditions and angles.
