#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[2,4,5,6,7,9,11]]
y = dataset.iloc[:,1]

y = y[X['Embarked'].notnull()].values
X = X[X['Embarked'].notnull()].values


#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X[:,2])
X[:,2] = imputer.transform(X[:,2])

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
  
onc = OneHotEncoder(categorical_features = [12])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]
  
#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#classification model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X,y)

#testing data
test_data = pd.read_csv('test.csv')
X_test = test_data.iloc[:,[1,3,4,5,6,8,10]].values

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X_test[:,2])
X_test[:,2] = imputer.transform(X_test[:,2])

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X_test[:,5])
X_test[:,5] = imputer.transform(X_test[:,5])


#categorical data handling
l=[0,1,3,4,6]

for j in l :
     X_test[:,j] = labelencoder.fit_transform(X_test[:,j].astype(str))

X_test = X_test.astype(float)
     

onc = OneHotEncoder(categorical_features = [0])
onc.fit(X_test)
X_test = onc.fit_transform(X_test).toarray()
X_test= X_test[:,1:]

onc = OneHotEncoder(categorical_features = [4])
onc.fit(X_test)
X_test = onc.fit_transform(X_test).toarray()
X_test= X_test[:,1:]

onc = OneHotEncoder(categorical_features = [12])
onc.fit(X_test)
X_test = onc.fit_transform(X_test).toarray()
X_test= X_test[:,1:]

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

y_test = classifier.predict(X_test)


