#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train multiple data using Autokeras

import pandas as pd
import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np

# Read SVM saved data(energy data)
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

# Read SVM saved data(volume data)
data2 = np.load('****.npz')
train_x2 = data2['train_x']
train_y2 = data2['train_y']
test_x2 = data2['test_x']
test_y2 = data2['test_y']

# Modify data format
train_y2 = train_y2[:,np.newaxis]
train_xy2 = np.concatenate((train_x2,train_y2),axis=1)
test_y2 = test_y2[:,np.newaxis]
test_xy2 = np.concatenate((test_x2,test_y2),axis=1)
pd_train_xy2 = pd.DataFrame(train_xy2)
column_names2 = ['column'+ str(i) for i in range(len(test_x2[0])) ]
column_names2.append('energy')
pd_train_xy2.columns = np.array(column_names2)
pd_test_xy2 = pd.DataFrame(test_xy2)
pd_test_xy2.columns = np.array(column_names2)
train_file_path2 = 'train2.csv'
test_file_path2 = 'eval2.csv'
pd_train_xy2.to_csv(train_file_path2, index=False)
pd_test_xy2.to_csv(test_file_path2, index=False)

# Initialize the multi with multiple inputs and outputs.
reg = ak.AutoModel(
    inputs=[ak.StructuredDataInput(), ak.StructuredDataInput()],
    outputs=[
        ak.RegressionHead(loss='logcosh'),
        ak.RegressionHead(loss="logcosh"),
    ],
    overwrite=True,
    max_trials=15,
)
# Fit the model with prepared data.
reg.fit(
    [train_x1, train_x2],
    [train_y1, train_y2],
    validation_data=(
        [test_x1, test_x2],
        [test_y1, test_y2],
    ),
    epochs=300,
)
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