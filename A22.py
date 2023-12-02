from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def datasetr(nf=1):
    datax = []
    datay = []
    with open("A2-ring_separable.txt") as f:
        data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
        for row in data:
            datax.append(row[:nf])
            datay.append(row[nf])
    return datax, datay

def testsetr(nf=1):
    datax = []
    datay = []
    with open("A2-ring-test.txt") as f:
        data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
        for row in data:
            datax.append(row[:nf])
            datay.append(row[nf])
    return datax, datay
            

datax, datay = datasetr(nf=2)
testx, testy = testsetr(nf=2)

def BP():
    model = Sequential()
    model.add(Dense(12, input_shape=(2,), activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(datax, datay, epochs=100, verbose=0)
    y_pred, accuracy = model.evaluate(testx, testy)
    y_pred = model.predict(testx)
    y_pred = [i[0] for i in y_pred]
    print("Mean squared error: {}".format(mean_squared_error(testy,y_pred)))


def MLR():
    regr = linear_model.LinearRegression()
    regr.fit(datax, datay)
    y_pred_scaled = regr.predict(testx)
    y_pred = list(y_pred_scaled)
    print("Mean squared error: {}".format(mean_squared_error(testy,y_pred)))

BP()
MLR()