'''
Panagiotis Christakakis
'''

''' Section(1) - Library Imports '''

import pandas as pd
import sklearn
import numpy as np
import keras
from pandas.core.arrays.categorical import factorize

''' Section(2) - Data Import & Anlysis '''

#Importing Data, Column Headings
fileName = 'Dataset2Use_Assignment1.xlsx'
sheetName = 'Total'
try:
  sheetValues = pd.read_excel(fileName, sheetName)
  print('.. successful parsing of file:', fileName)
  print('Column Headings:')
  print(sheetValues.columns)
except FileNotFoundError:
  print(FileNotFoundError)

#NA values for each feature, Type of variable for each Column
print("\n", sheetValues.isna().sum())
print("\n", sheetValues.info(), "\n")

#Factorize Output Categorical Values
inputData = sheetValues[sheetValues.columns[:-2]].values
outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)

#Paradigms, Features, Distribution of Class Labels
print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')
print(' ... the distribution for the available class lebels is:')
for classIdx in range(0, len(np.unique(outputData))):
 tmpCount = sum(outputData == classIdx)
 tmpPercentage = tmpCount/len(outputData)
 print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(','{:.2f}'.format(tmpPercentage), '%)')

''' Section(3) - Train_Test Split '''

# Split Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, test_size = 0.25, stratify = outputData, shuffle = True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

number_of_training_samples_train = X_train.shape[0]
non_healthy_counter_train = (y_train == 1).sum()
print("Number of training samples in TRAIN: ", number_of_training_samples_train, "and number of Non-Healthy companies in TRAIN: ", non_healthy_counter_train, "\n")

number_of_training_samples_test = X_test.shape[0]
non_healthy_counter_test = (y_test == 1).sum()
print("Number of training samples in TEST: ", number_of_training_samples_test, "and number of Non-Healthy companies in TEST: ", non_healthy_counter_test, "\n")

''' Section(4) - Model Fitting '''
#In order to run each model put a comment with (#) in front 
#of (''') at the start and at the end of each model section

# Section(4.1) - Linear Discriminant Analysis
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver = "svd")
lda.fit(X_train, y_train)
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)
'''

# Section(4.2) - Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = "lbfgs")
logreg.fit(X_train, y_train)
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
'''

# Section(4.3) - Decision Trees
'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 4)
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
'''

# Section(4.4) - k-Nearest Neighbors
'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
'''

# Section(4.5) - Naïve Bayes
'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
'''

# Section(4.6) - Support Vector Machines
# Hyper-parameter tuning was done after a loop
# process with GridSearchCV of a totaling 125 fits
'''
from sklearn.svm import SVC
svm = SVC(C = 0.1, gamma = 1, kernel = "rbf")
svm.fit(X_train, y_train)
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)
'''

# Section(4.7) - Neural Networks
#The dataset we're trying to classify is imbalanced.
#Class_0 to Class_1 ratio is 43 to 1. So even a neural
#network this big, can't even overfit to some Class_1 
#training samples. Training set resampling is required
#'''
CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(512, input_dim = X_train.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(256, activation='relu'))
CustomModel.add(keras.layers.Dense(128, activation='relu'))
CustomModel.add(keras.layers.Dense(64, activation='relu'))
CustomModel.add(keras.layers.Dense(32, activation='relu'))
CustomModel.add(keras.layers.Dense(16, activation='relu'))
CustomModel.add(keras.layers.Dense(8, activation='relu'))
CustomModel.add(keras.layers.Dense(4, activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
#CustomModel.summary()
CustomModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CustomModel.fit(X_train, keras.utils.np_utils.to_categorical(y_train), batch_size = 512, epochs=100, verbose=1)

y_pred_train = np.argmax((CustomModel.predict(X_train) > 0.5).astype("int32"), axis=1)
y_pred_test = np.argmax((CustomModel.predict(X_test) > 0.5).astype("int32"), axis=1)
#'''

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_test))

''' Section(5) - Metrics for Train_Test '''

# Calculate Accuracy, Precision, Recall, F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro') 
rec_test = recall_score(y_test, y_pred_test, average ='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')
# print the scores
print('Accuracy scores of <ModelName> classifier are:', 'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of <ModelName> classifier are:', 'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of <ModelName> classifier are:', 'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of <ModelName> classifier are:', 'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

# Calculate TP, TN, FP, FN
from sklearn.metrics import confusion_matrix

#Calculatin TP,TN,FP,FN of training set
tp_train, fp_train, fn_train, tn_train = confusion_matrix(y_train, y_pred_train).ravel()
print(tp_train, fp_train, fn_train, tn_train)

#Calculatin TP,TN,FP,FN of test set
tp_test, fp_test, fn_test, tn_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
print(tp_test, fp_test, fn_test, tn_test)

#Percentage Conditions of Models
percantage_bankrupt = tn_test / np.count_nonzero(y_test == 1)
percantage_non_bankrupt = tp_test / np.count_nonzero(y_test == 0)
print(percantage_bankrupt)
print(percantage_non_bankrupt)

''' Section(6) - Training Set Resampling '''

from imblearn.under_sampling import RandomUnderSampler
#Undersampling the majority class in order to have 3:1 ratio
rus = RandomUnderSampler(sampling_strategy=1/3)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

number_of_training_samples_train = X_train_rus.shape[0]
non_healthy_counter_train = (y_train_rus == 1).sum()
print("Number of training samples in TRAIN: ", number_of_training_samples_train, "and number of Non-Healthy companies in TRAIN: ", non_healthy_counter_train, "\n")

number_of_training_samples_test = X_test.shape[0]
non_healthy_counter_test = (y_test == 1).sum()
print("Number of training samples in TEST: ", number_of_training_samples_test, "and number of Non-Healthy companies in TEST: ", non_healthy_counter_test, "\n")

print(pd.value_counts(y_train_rus))

''' Section(7) - Model Fitting with Resampled Training Set '''
#In order to run each model put a comment with (#) in front 
#of (''') at the start and at the end of each model section

# Section(4.1) - Linear Discriminant Analysis
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver = "svd")
lda.fit(X_train_rus, y_train_rus)
y_pred_train = lda.predict(X_train_rus)
y_pred_test = lda.predict(X_test)
'''

# Section(4.2) - Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = "lbfgs")
logreg.fit(X_train_rus, y_train_rus)
y_pred_train = logreg.predict(X_train_rus)
y_pred_test = logreg.predict(X_test)
'''

# Section(4.3) - Decision Trees
'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 4)
clf.fit(X_train_rus, y_train_rus)
y_pred_train = clf.predict(X_train_rus)
y_pred_test = clf.predict(X_test)
'''

# Section(4.4) - k-Nearest Neighbors
'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_rus, y_train_rus)
y_pred_train = knn.predict(X_train_rus)
y_pred_test = knn.predict(X_test)
'''

# Section(4.5) - Naïve Bayes
'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_rus, y_train_rus)
y_pred_train = gnb.predict(X_train_rus)
y_pred_test = gnb.predict(X_test)
'''

# Section(4.6) - Support Vector Machines
'''
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_rus, y_train_rus)
y_pred_train = svm.predict(X_train_rus)
y_pred_test = svm.predict(X_test)
'''

# Section(4.7) - Neural Networks
#'''
import tensorflow as tf
from keras.utils import np_utils

custom_early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='accuracy',
                        min_delta=0.001,
                        patience=100,
                        mode='auto',
                        )

CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(64, input_dim = X_train_rus.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(16, activation='relu'))
CustomModel.add(keras.layers.Dense(16, activation='relu'))
CustomModel.add(keras.layers.Dense(32, activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
#CustomModel.add(keras.layers.Dense(1, activation='sigmoid'))
#CustomModel.summary()

CustomModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CustomModel.fit(X_train_rus, keras.utils.np_utils.to_categorical(y_train_rus), callbacks=[custom_early_stopping],
                batch_size = 50, epochs=200, verbose=1)

y_pred_train = np.argmax((CustomModel.predict(X_train_rus) > 0.5).astype("int32"), axis=1)
y_pred_test = np.argmax((CustomModel.predict(X_test) > 0.5).astype("int32"), axis =1)
#'''

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_test))

''' Section(8) - New Metrics for Train_Test '''

# Calculate Accuracy, Precision, Recall, F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_train = accuracy_score(y_train_rus, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train_rus, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train_rus, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average ='macro')
f1_train = f1_score(y_train_rus, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')
# print the scores
print('Accuracy scores of <ModelName> classifier are:', 'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of <ModelName> classifier are:', 'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of <ModelName> classifier are:', 'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of <ModelName> classifier are:', 'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

from sklearn.metrics import confusion_matrix

#Calculatin TP,TN,FP,FN of training set
tp_train, fp_train, fn_train, tn_train = confusion_matrix(y_train_rus, y_pred_train).ravel()
print(tp_train, fp_train, fn_train, tn_train)

#Calculatin TP,TN,FP,FN of test set
tp_test, fp_test, fn_test, tn_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
print(tp_test, fp_test, fn_test, tn_test)

#Percentage Conditions of Models
percantage_bankrupt = tn_test / np.count_nonzero(y_test == 1)
percantage_non_bankrupt = tp_test / np.count_nonzero(y_test == 0)
print(percantage_bankrupt)
print(percantage_non_bankrupt)
