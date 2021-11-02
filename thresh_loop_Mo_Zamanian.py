# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% class version: "FLD" algorithm, threshold loop for finding the optimum value


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
print('Head:\n', X_class_0[:5, ::])
print('Tail:\n', X_class_0[-5:, ::])
print('----------------------------------------------------------------------')
print('Class "1" info: Shape - Head - Tail')
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

# %% part 3 - calculating some points before search

u_mid = 0.5 * (u_class_0 + u_class_1)
u_low = 2*u_class_0 - u_mid
u_high = 2*u_class_1 - u_mid
u_list = [u_low, u_class_0, u_mid, u_class_1, u_high]
th_list = [np.dot(w.T, u) for u in u_list]
print(th_list)

# %% part 4 - loop over threshold --> np.linspace(-100, 100, 202)
# best_thresh = None; best_error = 1
# for a_threshold in np.linspace(-100, 100, 202):
#     predictions = (np.sign(np.dot(w, X_training_dataset.T) + a_threshold) + 1) / 2
#     error_possibility_1 = sum(predictions != y_training_dataset)
#     error_possibility_2 = sum((1 - predictions) != y_training_dataset)
#     rel_error_1 = error_possibility_1 / len(y_training_dataset)
#     rel_error_2 = error_possibility_2 / len(y_training_dataset)
#
#     # print('for thresh =', a_threshold, ', rel. error is:', min(rel_error_1, rel_error_2))
#
#     if min(rel_error_1, rel_error_2) < best_error:
#         best_thresh = a_threshold
#         best_error = min(rel_error_1, rel_error_2)
#
# print('best_thresh:', best_thresh, ', best_error:', best_error)


# %% part 4 - loop over threshold --> np.linspace(-1, 1, 2002)

# best_thresh = None; best_error = 1
# for a_threshold in np.linspace(-1, 1, 2002):
#     predictions = (np.sign(np.dot(w, X_training_dataset.T) + a_threshold) + 1) / 2
#     error_possibility_1 = sum(predictions != y_training_dataset)
#     error_possibility_2 = sum((1 - predictions) != y_training_dataset)
#     rel_error_1 = error_possibility_1 / len(y_training_dataset)
#     rel_error_2 = error_possibility_2 / len(y_training_dataset)
#
#     # print('for thresh =', a_threshold, ', rel. error is:', min(rel_error_1, rel_error_2))
#
#     if min(rel_error_1, rel_error_2) < best_error:
#         best_thresh = a_threshold
#         best_error = min(rel_error_1, rel_error_2)
#
# print('best_thresh:', best_thresh, ', best_error:', best_error)


# %% part 4 - loop over threshold --> np.linspace(-0.01, 0.01, 4002)

best_thresh = None; best_error = 1
for a_threshold in np.linspace(-0.01, 0.01, 4002):
    predictions = (np.sign(np.dot(w, X_training_dataset.T) + a_threshold) + 1) / 2
    error_possibility_1 = sum(predictions != y_training_dataset)
    error_possibility_2 = sum((1 - predictions) != y_training_dataset)
    rel_error_1 = error_possibility_1 / len(y_training_dataset)
    rel_error_2 = error_possibility_2 / len(y_training_dataset)

    # print('for thresh =', a_threshold, ', rel. error is:', min(rel_error_1, rel_error_2))

    if min(rel_error_1, rel_error_2) < best_error:
        best_thresh = a_threshold
        best_error = min(rel_error_1, rel_error_2)

print('best_thresh:', best_thresh, ', best_error:', best_error)

# %% part 5 - calculate confusion matrix for best threshold

# section 1 - analyze the order of labels

best_thresh = 0.002211947013246689
predictions = (np.sign(np.dot(w, X_training_dataset.T) + best_thresh) + 1) / 2
error_possibility_1 = sum(predictions != y_training_dataset)
error_possibility_2 = sum((1 - predictions) != y_training_dataset)
rel_error_1 = error_possibility_1 / len(y_training_dataset)
rel_error_2 = error_possibility_2 / len(y_training_dataset)

#  section 2 

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


tp = np.dot(y_training_dataset, new_preds)
tn = np.dot(1-y_training_dataset, 1-new_preds)

p = sum(y_training_dataset)  # p = tp + fn
n = len(y_training_dataset) - p  # n = tn + fp

fp = n - tn; fn = p - tp
print('tp = ; tn = ; fp = ; fn = ;',
      tp, tn, fp, fn)


