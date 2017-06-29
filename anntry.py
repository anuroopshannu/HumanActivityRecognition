import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, :-2].values
Y_train = train.iloc[:, 562].values
X_test = test.iloc[:, :-2].values
Y_test = test.iloc[:, 562].values


#
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import one_hot

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.fit_transform(Y_test)
Y_test = one_hot(Y_test)
Y_train = one_hot(Y_train)

import keras
from keras.models import Sequential
from keras.layers import Dense
def build():
    seq = Sequential()
    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform', input_dim = 561))
    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform'))
    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform'))
    seq.add(Dense(units = 6, activation = 'softmax', kernel_initializer = 'uniform'))
    seq.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return seq

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold

#kfold = KFold(n_splits=10, shuffle=True)

classifier = KerasClassifier(build_fn = build, batch_size = 32, nb_epoch= 20)
csv = cross_val_score(estimator= classifier, X = X_train, y = Y_train, cv = 10, n_jobs=-1)

mean = csv.mean()
std = csv.std()

classifier.fit(X_train, Y_train)
print("Accuracy: {}%\n".format(classifier.score(X_test, Y_test) *100))

classifier.predict(X_test)


        