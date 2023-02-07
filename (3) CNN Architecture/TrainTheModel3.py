'''
Panagiotis Christakakis
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

#define some parametes related to specific training problem
batch_size = 64
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print("CIFAR10 dataset has been imported")

#Creation of validiation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=1)
print("Validation Set has been created")

classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                    "Dog", "Frog", "Horse", "Ship", "Truck"]

print("We have: ", x_train.shape[0], " TRAINING paradigms of size: ", x_train.shape, "\n")
print("We have: ", x_val.shape[0], " VALIDATION paradigms of size: ", x_val.shape, "\n")
print("We have: ", x_test.shape[0], " TESTING paradigms of size: ", x_test.shape, "\n")

print("Training Set: Νumber of elements for each class")
unique_train, counts_train = np.unique(y_train, return_counts=True)
print(dict(zip(unique_train, counts_train)), "\n")

print("Validation Set: Νumber of elements for each class")
unique_val, counts_val = np.unique(y_val, return_counts=True)
print(dict(zip(unique_val, counts_val)), "\n")

print("Testing Set: Νumber of elements for each class")
unique_test, counts_test = np.unique(y_test, return_counts=True)
print(dict(zip(unique_test, counts_test)), "\n")

#plot randomly 4 images for each class

class_to_demonstrate = 0
while (sum(y_train == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_train == class_to_demonstrate)

    # create new plot window
    plt.figure()

    # plot 4 random images
    plt.subplot(221)
    plt.imshow(x_train[tmp_idxs_to_use[0][randrange(counts_train[class_to_demonstrate])], :, :, :])
    plt.subplot(222)
    plt.imshow(x_train[tmp_idxs_to_use[0][randrange(counts_train[class_to_demonstrate])], :, :, :])
    plt.subplot(223)
    plt.imshow(x_train[tmp_idxs_to_use[0][randrange(counts_train[class_to_demonstrate])], :, :, :])
    plt.subplot(224)
    plt.imshow(x_train[tmp_idxs_to_use[0][randrange(counts_train[class_to_demonstrate])], :, :, :])
    tmp_title = 'Class: ' + str(classes[class_to_demonstrate])
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# here we define the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#compiling the structure
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
print("CNN model has been compiled. Proceeding to fitting.")

# print model summary
model.summary()

# fit model parameters
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))

#plot train-val loss & accuracy for epochs
def plot_metric(history, metric):
      train_metrics = history.history[metric]
      val_metrics = history.history['val_'+metric]
      epochs = range(1, len(train_metrics) + 1)
      plt.plot(epochs, train_metrics)
      plt.plot(epochs, val_metrics)
      plt.title('Training and Validation '+ metric)
      plt.xlabel("Epochs")
      plt.ylabel(metric)
      plt.legend(["train_"+metric, 'val_'+metric])
      plt.show()

plot_metric(history, 'loss')
plot_metric(history, 'accuracy')

# saving the trained model
model_name = 'CIFAR-10_CNN.h5'
model.save(model_name)
