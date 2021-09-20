#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using DNN/ResNet

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,BatchNormalization,Input,add # , LSTM, Activation
from keras.optimizers import Adam, Adadelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import time

# Read SVM saved data
data = np.load('****.npz')

# DNN model training
time_start1=time.time()
train_x1 = data['train_x']
train_y1 = data['train_y']
test_x1 = data['test_x']
test_y1 = data['test_y']
model1 = Sequential()
model1.add(Dense(53, init='uniform', activation='tanh',input_dim=train_x1.shape[1]))
model1.add(BatchNormalization())
model1.add(Dense(128, activation='tanh'))
model1.add(BatchNormalization())
model1.add(Dense(256, activation='tanh'))
model1.add(BatchNormalization())
model1.add(Dropout(0.05))
model1.add(Dense(32, activation='tanh'))
model1.add(BatchNormalization())
model1.add(Dropout(0.05))
model1.add(Dense(1, activation='linear'))
adamoptimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model1.compile(optimizer=adamoptimizer1, loss='mae')
history1= model1.fit(train_x1, train_y1, epochs=300, batch_size=256,validation_data=(test_x1, test_y1))
predict_y1 = model1.predict(test_x1, batch_size=1)
loss1=model1.evaluate(test_x1, test_y1) # , batch_size=10)
time_end1=time.time()
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('DNN Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y1,predict_y1,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y1,test_y1)
plt.grid(True)
plt.show()


# ResNet model training
time_start2=time.time()
train_x2 = data['train_x']
train_y2 = data['train_y']
test_x2 = data['test_x']
test_y2 = data['test_y']
input_layer = Input(shape=(train_x2.shape[1], ))
dense1 = Dense(53, init='uniform', activation='tanh')(input_layer)
dense1 = BatchNormalization()(dense1)
dense2 = Dense(128, activation='tanh')(dense1)
dense2 = BatchNormalization()(dense2)
dense3 = Dense(128, activation='tanh')(dense1)
dense3 = BatchNormalization()(dense3)
dense3 = Dropout(0.05)(dense3)
dense4 = Dense(53,  activation='tanh')(dense3)
dense4 = BatchNormalization()(dense4)
z1 = add([dense1, dense4])
dense5 = Dense(32, activation='tanh')(z1)
dense5 = BatchNormalization()(dense5)
dense5 = Dropout(0.05)(dense5)
out_layer = Dense(1,  activation='linear')(dense5)
model2 = Model(inputs=input_layer, outputs=out_layer)
adamoptimizer2 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model2.compile(optimizer=adamoptimizer2, loss='mae')
history2= model2.fit(train_x2, train_y2, epochs=300, batch_size=256,validation_data=(test_x2, test_y2))
predict_y2 = model2.predict(test_x2, batch_size=1)
loss=model2.evaluate(test_x2, test_y2)
time_end2=time.time()
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('ResNet Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y2,predict_y2,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y2,test_y2)
plt.grid(True)
plt.show()


# Autokeras best model for export
# Use DNN to achieve the best model
time_start3=time.time()
train_x3 = data['train_x']
train_y3 = data['train_y']
test_x3 = data['test_x']
test_y3 = data['test_y']
model3 = Sequential()
model3.add(Dense(53, init='uniform', activation='relu',input_dim=train_x3.shape[1]))
model3.add(BatchNormalization())
model3.add(Dense(32, activation='relu'))
model3.add(Dense(1024, activation='relu'))
model3.add(Dropout(0.25))
model1.add(Dense(32, activation='relu'))
model3.add(Dense(1, activation='linear'))
adamoptimizer3 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model3.compile(optimizer=adamoptimizer3, loss='mae')
history3= model3.fit(train_x3, train_y3, epochs=300, batch_size=256,validation_data=(test_x3, test_y3))
predict_y3 = model3.predict(test_x3, batch_size=1)
loss3=model3.evaluate(test_x3, test_y3)
time_end3=time.time()
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('AutoKeras Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y3,predict_y3,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y3,test_y3)
plt.grid(True)
plt.show()


# Summarize the results of the three models
epochs1 = len(history1.history['loss'])
plt.plot(range(epochs1), history1.history['val_loss'], label='DNN_Loss')
epochs2 = len(history2.history['loss'])
plt.plot(range(epochs2), history2.history['val_loss'], label='ResNet_loss')
epochs3 = len(history3.history['loss'])
plt.plot(range(epochs3), history3.history['val_loss'], label='AutoKeras_loss')
plt.ylabel('MAE Loss', fontname='Arial Narrow', size=16)
plt.xlabel('Epoch', fontname='Arial Narrow', size=16)
plt.ylim(0, 1.0)
plt.legend()
plt.show()

print('DNN totally cost time:',time_end1-time_start1)
print('ResNet totally cost time:',time_end2-time_start2)
print('Autokeras totally cost time:',time_end3-time_start3)
model1_RMSE = 1 - (mean_squared_error(test_y1,predict_y1)/np.var(test_y1))
print(model1_RMSE)
model1_MAE=mean_absolute_error(test_y1,predict_y1)
print(model1_MAE)
model2_RMSE = 1 - (mean_squared_error(test_y2,predict_y2)/np.var(test_y2))
print(model2_RMSE)
model2_MAE=mean_absolute_error(test_y2,predict_y2)
print(model2_MAE)
model3_RMSE = 1 - (mean_squared_error(test_y3,predict_y3)/np.var(test_y3))
print(model3_RMSE)
model3_MAE=mean_absolute_error(test_y3,predict_y3)
print(model3_MAE)