#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using Autokeras

import pandas as pd
import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np

# Read SVM saved data
data1 = np.load('****.npz')
train_x1 = data1['train_x']
train_y1 = data1['train_y']
test_x1 = data1['test_x']
test_y1 = data1['test_y']

# Modify data format
train_y1 = train_y1[:,np.newaxis]
train_xy1 = np.concatenate((train_x1,train_y1),axis=1)
test_y1 = test_y1[:,np.newaxis]
test_xy1 = np.concatenate((test_x1,test_y1),axis=1)
pd_train_xy1 = pd.DataFrame(train_xy1)
column_names1 = ['column'+ str(i) for i in range(len(test_x1[0])) ]
column_names1.append('energy')
pd_train_xy1.columns = np.array(column_names1)
pd_test_xy1 = pd.DataFrame(test_xy1)
pd_test_xy1.columns = np.array(column_names1)
train_file_path1 = 'train1.csv'
test_file_path1 = 'eval1.csv'
pd_train_xy1.to_csv(train_file_path1, index=False)
pd_test_xy1.to_csv(test_file_path1, index=False)

# Initialize the structured data regressor
reg = ak.StructuredDataRegressor(
    overwrite=True, loss='logcosh',
    max_trials=10) # It tries 10 different models.
# Feed the structured data regressor with training data.
history=reg.fit(
    # The path to the train.csv file.
    train_file_path1,
    # The name of the label column.
    'energy',
    validation_data=(test_x1, test_y1),
    epochs=300)
# Predict with the best model.
predict_y = reg.predict(test_file_path1)
# Evaluate the best model with testing data.
print(reg.evaluate(test_file_path1,'energy'))

# Draw the result diagram
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('AutoKeras Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y1,predict_y,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y1,test_y1)
plt.grid(True)
plt.show()

#Model export
model = reg.export_model()
file_name = 'AutoKeras_***'
model.save(file_name + '.h5')