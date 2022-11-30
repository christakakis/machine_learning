'''
Panagiotis Christakakis
AIDA, Dept. of Apllied Informatics, UoM
aid23004@uom.edu.gr
ID: aid23004
'''

#library imports
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from random import randrange

model_name = 'CIFAR-10_CNN.h5'

# loading a trained model & use it over test data
loaded_model = keras.models.load_model(model_name)
print("Model was loaded successfully")

#loading again x_test, y_test
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                    "Dog", "Frog", "Horse", "Ship", "Truck"]

# calculate some common performance scores
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_test_predictions_vectorized = loaded_model.predict(x_test)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
print("Model prediction was finished successfully")

y_test_argmax = np.argmax(y_test, axis=1)

#Classification Report
print("Classification Report: \n", classification_report(y_test_argmax, y_test_predictions))

#Confusion Matrix
print("Confusion Matrix: \n", confusion_matrix(y_test_argmax, y_test_predictions))

print("Predictions: Νumber of elements for each class")
unique_test_pred, counts_test_pred = np.unique(y_test_predictions, return_counts=True)
print(dict(zip(unique_test_pred, counts_test_pred)), "\n")

#plot randomly 4 images for each prediction class
class_to_demonstrate = 0
while (sum(y_test_predictions == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # create new plot window
    plt.figure()

    # plot 4 random images
    plt.subplot(221)
    plt.imshow(x_test[tmp_idxs_to_use[0][randrange(counts_test_pred[class_to_demonstrate])], :, :, :])
    plt.subplot(222)
    plt.imshow(x_test[tmp_idxs_to_use[0][randrange(counts_test_pred[class_to_demonstrate])], :, :, :])
    plt.subplot(223)
    plt.imshow(x_test[tmp_idxs_to_use[0][randrange(counts_test_pred[class_to_demonstrate])], :, :, :])
    plt.subplot(224)
    plt.imshow(x_test[tmp_idxs_to_use[0][randrange(counts_test_pred[class_to_demonstrate])], :, :, :])
    tmp_title = 'Class: ' + str(classes[class_to_demonstrate])
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1
