# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% class version: "FLD" algorithm, trying different values of threshold for classification


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
print('Class "0" info: Shape - Head - Tail')
print('Shape:', X_class_0.shape)
# print('Head:\n', X_class_0[:5, ::])
# print('Tail:\n', X_class_0[-5:, ::])
print('----------------------------------------------------------------------')
print('Class "1" info: Shape - Head - Tail')
print('Shape:', X_class_1.shape)
# print('Head:\n', X_class_1[:5, ::])
# print('Tail:\n', X_class_1[-5:, ::])
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
while True:
    print('To exit, press "0".')
    threshold = input('Please enter threshold (hint: a small negative number)')
    if threshold == '0': break
    threshold = float(threshold)

    # %% part 3 - prediction
    
    predictions = (np.sign(np.dot(w, X_training_dataset.T) + threshold) + 1) / 2
    error_vec_1 = sum(predictions != y_training_dataset)
    error_vec_2 = sum((1 - predictions) != y_training_dataset)
    rel_error_1 = error_vec_1 / len(y_training_dataset)
    rel_error_2 = error_vec_2 / len(y_training_dataset)
    if rel_error_1 < rel_error_2:
        new_preds = predictions
    else:
        new_preds = 1 - predictions
    tp = np.dot(y_training_dataset, new_preds)
    tn = np.dot(1-y_training_dataset, 1-new_preds)

    # %% part 4 - error 
    
    print('rel. class-1 Error = ', tp / len(y_training_dataset),
          ', rel. class-0 Error= ', tn / len(y_training_dataset))
    

