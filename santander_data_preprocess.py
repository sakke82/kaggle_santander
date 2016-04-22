'''
author: Sakari Hakala, sakari.hakala@gmail.com
Script to preprocess data for Kaggle Competition: Santander Customer Satisfaction
https://www.kaggle.com/c/santander-customer-satisfaction
'''


import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# load data

train = pd.read_csv('./data/train.csv', index_col = 0)
test = pd.read_csv('./data/test.csv', index_col = 0)

print 'train set:', train.shape
print 'test set:', test.shape

# drop rows with zero variance
remove = [] 
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis = 1, inplace = True)
test.drop(remove, axis = 1, inplace = True)
print 'number of removed 0-variance columns:', len(remove)

# remove duplicate columns
remove_2 = []
columns = train.columns
# run through all columns
for i in range(len(columns) - 1):
    this_col = train[columns[i]].values
    # and compare to all columns to the right from 'this_col'
    for j in range(i+1, len(columns)):
        if np.array_equal(this_col, train[columns[j]].values):
            remove_2.append(columns[j])

train.drop(remove_2, axis = 1, inplace = True)
test.drop(remove_2, axis = 1, inplace = True)
print 'number of removed duplicated columns:', len(remove_2)

# separate TARGET from train
y = train.TARGET
train.drop('TARGET', axis = 1, inplace = True)

print 'train set:', train.shape
print 'test set:', test.shape

# put train and test sets together
all_data = train.append(test)
# normalize the data and separate again
all_data = normalize(all_data, axis=0)
train_norm = all_data[:train.shape[0]]
test_norm = all_data[train.shape[0]:]
# and calculate PCA
pca = PCA(n_components = 2)
train_norm_projected = pca.fit_transform(train_norm)
test_norm_projected = pca.transform(test_norm)

# add new features to our sets
# zero count
train['0_count'] = (train==0).sum(axis=1)
test['0_count'] = (test==0).sum(axis=1)
# and principle components
train['PCA1'] = train_norm_projected[:,0]
train['PCA2'] = train_norm_projected[:,1]
test['PCA1'] = test_norm_projected[:,0]
test['PCA2'] = test_norm_projected[:,1]
# and add TARGET back to train set
train['TARGET'] = y
# and now we got
print 'train set:', train.shape
print 'test set:', test.shape

train.to_csv('./data/train_preprocessed.csv')
test.to_csv('./data/test_preprocessed.csv')
