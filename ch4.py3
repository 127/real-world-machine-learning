import random as r
import numpy as np
import math as m
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

features = rng.rand(100,5)
# print(features)
target = rng.rand(100) > 0.5
# print(target)

N = features.shape[0]
# print(N)
N_train = m.floor(0.7 * N)
# print(N_train)

# Randomize index
# Note: sometimes you want to retain the order in the dataset and skip this step
# E.g. in the case of time-based datasets where you want to test on 'later' instances
idx = np.random.permutation(N)
# print(idx.shape[0])

# Split index
idx_train = idx[:N_train]
idx_test = idx[N_train:]

# Break your data into training and testing subsets
features_train = features[idx_train,:]
# print(features[idx_train,:].shape)
target_train = target[idx_train]
# print(target_train)
features_test = features[idx_test,:]
target_test = target[idx_test]
# print()
# Build, predict, evaluate (to be filled out)
# model = train(features_train, target_train)
# preds_test = predict(model, features_test)
# accuracy = evaluate_acc(preds_test, target_test)

print(features_train.shape)
print(features_test.shape)
print(target_train.shape)
print(target_test.shape)

N = features.shape[0]
K = 10 # number of folds

preds_kfold = np.empty(N)
folds = np.random.randint(0, K, size=N)

for idx in np.arange(K):

    # For each fold, break your data into training and testing subsets
    features_train = features[folds != idx,:]
    target_train = target[folds != idx]
    features_test = features[folds == idx,:]
    
    # Print the indices in each fold, for inspection
    print(np.nonzero(folds == idx)[0])

    # Build and predict for CV fold (to be filled out)
    # model = train(features_train, target_train)
    # preds_kfold[folds == idx] = predict(model, features_test)
    
# accuracy = evaluate_acc(preds_kfold, target)
# print(accuracy)