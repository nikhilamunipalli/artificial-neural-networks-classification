#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import torch

 

#importing dataset
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_test = test.iloc[:,[1,3,4,5,6,8,10]]
X = dataset.iloc[:,[2,4,5,6,7,9,11]]
y = dataset.iloc[:,1]

y = y[X['Embarked'].notnull()].values
X = X[X['Embarked'].notnull()]

X = X.append(X_test)
X= X.values

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X[:,2])
X[:,2] = imputer.transform(X[:,2])

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X[:,5])
X[:,5] = imputer.transform(X[:,5])

#categorical data handling
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

l=[0,1,3,4,6]

for j in l :
     X[:,j] = labelencoder.fit_transform(X[:,j].astype(str))

X = X.astype(float)

onc = OneHotEncoder(categorical_features = [0])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [4])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [10])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [18])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#splitting test and train data
X_test = X[889:,:]
X = X[:889,:]

'''#training testing data spit
from sklearn.model_selection import train_test_split
X_train,X_test1,y_train,y_test = train_test_split(X,y,test_size = 0.2)'''

#building the ANN model
classifier = Sequential()

#first layer of neural network
classifier.add(Dense(input_dim = 20 , output_dim = 30, init = 'uniform', activation = 'relu'))

#internal layers
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', metrics =['accuracy'],loss = 'binary_crossentropy' )
classifier.fit(X,y,batch_size = 5, nb_epoch = 100)

#predicting the results
y_pred = classifier.predict(X_test)
y_pred[y_pred>0.55] = 1
y_pred[y_pred<0.55] = 0













