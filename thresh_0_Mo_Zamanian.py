# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% class version: "FLD" algorithm assuming threshold is 0


import numpy as np
import pandas as pd

# %% part 1 - load dataset and print summary

training_dataset = pd.read_excel('spam.xlsx').values
X_training_dataset = training_dataset[::, :-1]
y_training_dataset = training_dataset[::, -1]

X_class_0_idx = np.where(y_training_dataset == 0)
X_class_1_idx = np.where(y_training_dataset == 1)
X_class_0 = X_training_dataset[X_class_0_idx]
X_class_1 = X_training_dataset[X_class_1_idx]

print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('Summary of the Dataset:')
print('Class "0" info: Shape')
print('Shape:', X_class_0.shape)
print('Head:\n', X_class_0[:5, ::])
print('Tail:\n', X_class_0[-5:, ::])
print('----------------------------------------------------------------------')
print('Class "1" info: Shape')
print('Shape:', X_class_1.shape)
print('Head:\n', X_class_1[:5, ::])
print('Tail:\n', X_class_1[-5:, ::])
print('----------------------------------------------------------------------')

# %% part 2 - perform Fisher Linear Discriminant (FLD)

# compute means of classes
u_class_0 = np.mean(X_class_0, 0)
u_class_1 = np.mean(X_class_1, 0)

# remove means from classes
X0_mean_red = X_class_0 - u_class_0
X1_mean_red = X_class_1 - u_class_1

# calculate covariance matrices
S0 = np.dot(X0_mean_red.T, X0_mean_red)
S1 = np.dot(X1_mean_red.T, X1_mean_red)
Sw = S0 + S1

# calculate slope (projector) and intercept (threshold)
w = np.dot(np.linalg.inv(Sw), (u_class_0 - u_class_1))
threshold = 0

# %% part 3 - prediction

print('----------------------------------------------------------------------')
print('FLD >> Slope and Intercept:')
print('Slope =', w, ', Intercept =', float(threshold))
print('----------------------------------------------------------------------')

predictions = (np.sign(np.dot(w, X_training_dataset.T) + threshold) + 1) / 2
error_possibility_1 = sum(predictions != y_training_dataset)
error_possibility_2 = sum((1 - predictions) != y_training_dataset)
rel_error_1 = error_possibility_1 / len(y_training_dataset)
rel_error_2 = error_possibility_2 / len(y_training_dataset)

# %% part 4 - error report

if rel_error_1 < rel_error_2:
    errorIndex = np.argwhere(predictions != y_training_dataset)
    new_preds = predictions
    errorPts = X_training_dataset[errorIndex]
    errorPts = np.squeeze(errorPts)

else:
    errorIndex = np.argwhere((1 - predictions) != y_training_dataset)
    new_preds = 1 - predictions
    errorPts = X_training_dataset[errorIndex]
    errorPts = np.squeeze(errorPts)

print('----------------------------------------------------------------------')
print('FLD >> Error:', 100 * min(rel_error_2, rel_error_1), '%.')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# %% part 5 - data exploration

print('Data Exploration:')
print('some samples of original class labels:\n', y_training_dataset[::10])
print('some samples of predictions:\n', new_preds[::10])


