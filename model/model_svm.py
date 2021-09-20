#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using SVM

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_absolute_error

# Optional:formationenergy.npz\volume.npz\energy.npz
data = np.load('***.npz',allow_pickle=True)
x=data["X"]
y=data["y"]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
time_start1=time.time()
clf = SVR()
clf.fit(train_x, train_y)
print(clf.score(test_x, test_y)) # Score method use RMSE
predict_y = clf.predict(test_x)
MAE=mean_absolute_error(test_y,predict_y)
print(MAE)
time_end1=time.time()
print('totally cost time:',time_end1-time_start1) # Model run time

# Draw the result diagram
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('Support Vector Machine Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y,predict_y,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y,test_y)
plt.grid(True)
plt.show()

# Save the split dataset for later models
np.savez('****.npz', train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)