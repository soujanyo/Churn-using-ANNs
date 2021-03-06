# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:36:05 2020

@author: MONALIKA P
"""
#Implimenting ANN using K cross validation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Working on dummy variables
df1=pd.get_dummies(dataset['Geography'],drop_first=True)
df2=pd.get_dummies(dataset['Gender'],drop_first=True)

dataset = pd.concat([df1,df2,dataset],axis=1)

#Removal of unwanted data from that dataset
dataset.drop('Geography',axis=1,inplace=True)
dataset.drop('Gender',axis=1,inplace=True)
dataset.drop('RowNumber',axis=1,inplace=True)
dataset.drop('CustomerId',axis=1,inplace=True)
dataset.drop('Surname',axis=1,inplace=True)

#obtaining  a matrix of features and prediction vector
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# evaluating ANN using K cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean=accuracies.mean()
variance=accuracies.std()
