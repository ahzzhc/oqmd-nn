#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using auto_ml

import numpy as np
from auto_ml import Predictor
import pandas as pd
import matplotlib.pyplot as plt
from auto_ml.utils_models import load_ml_model

# Read SVM saved data
data = np.load('****.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

# Modify data format
train_y = train_y[:,np.newaxis]
train_xy = np.concatenate((train_x,train_y),axis=1)
test_y = test_y[:,np.newaxis]
test_xy = np.concatenate((test_x,test_y),axis=1)

# Model training
pd_train_xy = pd.DataFrame(train_xy)
column_names = ['column'+ str(i) for i in range(len(test_x[0])) ]
column_names.append('volume')
pd_train_xy.columns = np.array(column_names)
pd_test_xy = pd.DataFrame(test_xy)
pd_test_xy.columns = np.array(column_names)
column_descriptions = {
    'volume': 'output'
}
ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)
ml_predictor.train(pd_train_xy, model_names='DeepLearningRegressor')
score = ml_predictor.score(pd_test_xy, pd_test_xy.volume)

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()
trained_model = load_ml_model(file_name)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predict_y = trained_model.predict(pd_test_xy)
print(score)

# Draw the result diagram
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.title('auto_ml Model', fontsize=20, pad=12)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=16)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=16)
plt.scatter(test_y,predict_y,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y,test_y)
plt.grid(True)
plt.show()