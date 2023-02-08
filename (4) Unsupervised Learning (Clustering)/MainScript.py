'''
Panagiotis Christakakis
'''

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import unique
from sklearn import cluster
from tensorflow import keras
from random import randrange
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from sklearn import cluster, metrics
from keras.datasets import fashion_mnist
from sklearn.mixture import GaussianMixture
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation

#define a performance evaluation function
from sklearn import metrics
def performance_score(input_values, cluster_indexes):
    try:
        silh_score = metrics.silhouette_score(input_values.reshape(-1, 1), cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        #print( ' ... -1: incorrect, 0: overlapping, +1: highly dense clusts.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score = metrics.calinski_harabasz_score(input_values.reshape(-1, 1), cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        #print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = metrics.davies_bouldin_score(input_values.reshape(-1, 1), cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index score.')
        db_score = -999

    try:
        adj_mutual_score = metrics.adjusted_mutual_info_score(input_values, cluster_indexes)
        print(' .. Adjusted Mutual Info score is {:.2f}'.format(adj_mutual_score))
    except:
        print(' .. Warning: could not calculate Adjusted Mutual Info score.')
        adj_mutual_score = -999

    return silh_score, ch_score, db_score, adj_mutual_score

#define some parametes related to specific training problem
batch_size = 64
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# scale pixels
# convert from integers to floats
train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')
# normalize to range 0-1
x_train = train_norm / 255.0
x_test = test_norm / 255.0

# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same',activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

#train the model
history = model.fit(X_train, X_train, epochs = 3, batch_size = 64, 
                    validation_data=(X_validate, X_validate), verbose=1)

#plot train-val loss & accuracy for epochs
def plot_metric(history, loss):
      train_loss = history.history[loss]
      
      val_loss = history.history['val_'+loss]
      epochs = range(1, len(train_loss) + 1)
      plt.plot(epochs, train_loss)
      plt.plot(epochs, val_loss)
      plt.title('Training and Validation '+ loss)
      plt.xlabel("Epochs")
      plt.ylabel(loss)
      plt.legend(["train_"+loss, 'val_'+loss])
      plt.show()

plot_metric(history, 'loss')

# do some plotting for the reconstructed images
import matplotlib.pyplot as plt
restored_testing_dataset = model.predict(x_test)
# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

#exctract the encoder block
encoder = K.function([model.layers[0].input],[model.layers[4].output])

#convert images to projected data
test_encoded_images = encoder([x_test])[0].reshape(-1,7*7*7)

#K MEANS CLUSTERING

#use projections to cluster the images
for numOfClust in range (3,12):
  print('Currently testing', str(numOfClust), 'number of clusters')
  mbkm = cluster.MiniBatchKMeans(n_clusters = numOfClust)
  mbkm.fit(test_encoded_images)
  clusterLabels = mbkm.labels_
  silh_score, ch_score, db_score, adj_mutual_score = performance_score(y_test, clusterLabels)

#use minibatch kmeans with 10 clusters
# Cluster the training set
mbkm = cluster.MiniBatchKMeans(n_clusters = 3)
mbkm.fit(test_encoded_images)
clusterLabels = mbkm.labels_

#performance scores & visualizations
fig = plt.figure(figsize=(20,20))
for clusterIdx in range(10):
    # cluster = cm[r].argmax()
    for c, val in enumerate(x_test[clusterLabels == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(clusterIdx))

#DBSCAN Clustering
eps_test = 1
while eps_test <= 2:
  for min_samples_test in range(1,10):
    print('Currently testing for eps: ', str(eps_test), 'and min_samples: ', str(min_samples_test))
    dbscan = cluster.DBSCAN(eps = eps_test, min_samples = min_samples_test)
    dbscan.fit(test_encoded_images)
    clusterLabels = dbscan.labels_
    clusters = unique(clusterLabels)
    silh_score, ch_score, db_score, adj_mutual_score = performance_score(y_test, clusterLabels)
    print(len(clusters))
  eps_test += 0.5

#use dbscan with best parameter
# Cluster the training set
dbscan = cluster.DBSCAN(eps = 1, min_samples = 9)
dbscan.fit(test_encoded_images)
clusterLabels = dbscan.labels_
silh_score, ch_score, db_score, adj_mutual_score = performance_score(y_test, clusterLabels)

#performance scores & visualizations
fig = plt.figure(figsize=(20,20))
for clusterIdx in range(10):
    # cluster = cm[r].argmax()
    for c, val in enumerate(x_test[clusterLabels == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(clusterIdx))

#OPTICS Clustering
eps_test = 0.5
while eps_test <= 2:
  for min_samples_test in range(1,8):
    print('Currently testing for eps: ', str(eps_test), 'and min_samples: ', str(min_samples_test))
    optics = cluster.OPTICS(eps = eps_test, min_samples = min_samples_test)
    optics.fit(test_encoded_images)
    clusterLabels = optics.labels_
    clusters = unique(clusterLabels)
    silh_score, ch_score, db_score, adj_mutual_score = performance_score(y_test, clusterLabels)
    print(len(clusters))
  eps_test += 0.5

#use dbscan with best parameter
# Cluster the training set
optics = cluster.OPTICS(eps = 0.5, min_samples = 7)
optics.fit(test_encoded_images)
clusterLabels = optics.labels_
silh_score, ch_score, db_score, adj_mutual_score = performance_score(y_test, clusterLabels)

#performance scores & visualizations
fig = plt.figure(figsize=(20,20))
for clusterIdx in range(10):
    # cluster = cm[r].argmax()
    for c, val in enumerate(x_test[clusterLabels == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(clusterIdx))