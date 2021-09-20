#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train multiple data using DNN

from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import Adam, RMSprop, Adadelta
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error

# Read data and split data
time_start1=time.time()
data = np.load('multitasking.npz')
X = data['X']
y = data['y'].T
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.9)

# Model training
model1 = Sequential()
model1.add(Dense(53, init='uniform', activation='relu',input_dim=train_x.shape[1]))
model1.add(BatchNormalization())
model1.add(Dense(128, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(128, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.05))
model1.add(Dense(32, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(32, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.05))
model1.add(Dense(8, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dense(2, activation='linear'))
adamoptimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model1.compile(optimizer=adamoptimizer1, loss='logcosh')
history1= model1.fit(train_x, train_y, epochs=300, batch_size=256)
predict_y1 = model1.predict(test_x, batch_size=1)

# Summarize the results
volume_MAE = mean_absolute_error(test_y.T[0].T,predict_y1.T[0].T)
print(volume_MAE)
energy_MAE = mean_absolute_error(test_y.T[1].T,predict_y1.T[1].T)
print(energy_MAE)
volume_RMSE = 1 - (mean_squared_error(test_y.T[0].T,predict_y1.T[0].T)/np.var(test_y.T[0].T))
print(volume_RMSE)
energy_RMSE = 1 - (mean_squared_error(test_y.T[1].T,predict_y1.T[1].T)/np.var(test_y.T[1].T))
print(energy_RMSE)
time_end1=time.time()
print('totally cost time:',time_end1-time_start1)