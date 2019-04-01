# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:03:58 2019

@author: Q466091
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
#import xgboost as xgb
import pickle
import os
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from datetime import datetime
import featuretools as ft

gc.enable()


train_path = 'data/train.csv'
test_path  = 'data/test.csv'
lgb_path = 'lgbm_models/'





train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train.drop(['ID_code'], axis=1, inplace=True)

orig_train = train


# Study variables
train['var_68'].describe()
print(train['var_68'].describe())

"""VAR 12
mean         14.023978
std           0.190059
min          13.434600
25%          13.894000
50%          14.025500
75%          14.164200
max          14.654500
"""

"""VAR 81
mean         14.719024
std           2.299567
min           7.586500
25%          13.214775
50%          14.844500
75%          16.340800
max          23.132400
"""

"""VAR 108
mean         14.224435
std           0.171091
min          13.729000
25%          14.098900
50%          14.226600
75%          14.361800
max          14.743000
"""

"""VAR 15
mean         14.573126
std           0.411711
min          13.076900
25%          14.262800
50%          14.574100
75%          14.874500
max          15.863300
"""

"""VAR 68
mean          5.018893
std           0.007186
min           4.993800
25%           5.014000
50%           5.019100
75%           5.024100
max           5.046900
"""

"""
epoch_datetime = pd.datetime(1900, 1, 1)
trf_var_68_s = (train['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
date_s68 = trf_var_68_s.map(datetime.fromordinal)
train['date68'] = date_s68
sorted_train_df68 = train.drop('var_68', axis=1).sort_values('date68')


trf_var_108_s = (train['var_108']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
date_s108 = trf_var_108_s.map(datetime.fromordinal)
train['date108'] = date_s108
sorted_train_df108 = train.drop(['date68', 'var_108'], axis=1).sort_values('date108')

trf_var_12_s = (train['var_12']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
date_s12 = trf_var_12_s.map(datetime.fromordinal)
train['date12'] = date_s12
sorted_train_df12 = train.drop(['date68', 'date108','var_12'], axis=1).sort_values('date12')
"""

def decomposeDate(date):
    
    month = date.month
    week = date.week
    day = date.day
    
    return (month, week, day)

smth = pd.cut(train['var_81'], 5).head()

print(pd.cut(train['var_81'], 5).head())
"""
bins_12 = ['1', '2', '3', '4', '5']
range_12 = pd.cut(train['var_12'], 5, labels = bins_12)
range_12 = range_12.to_frame()
range_12 = pd.get_dummies(range_12)

bins_81 = ['1', '2', '3', '4', '5', '6', '7']
range_81 = pd.cut(train['var_81'], 7, labels = bins_81)
range_81 = range_81.to_frame()
range_81 = pd.get_dummies(range_81)

train = pd.concat([train,range_12, range_81],axis = 1)
"""
# Var 68 maybe a date
"""
epoch_datetime = pd.datetime(1900, 1, 1)
trf_var_68_s = (train['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
date_s68 = trf_var_68_s.map(datetime.fromordinal)

train['date'] = date_s68
train['year'] = date_s68.map(lambda x: x.year)
train['month'] = date_s68.map(lambda x: x.month)
train['week'] = date_s68.map(lambda x: x.weekday())
train['day'] = date_s68.map(lambda x: x.day)

train.drop(['var_68', 'date'], axis=1).sort_values('date')
"""




# Explore unique values:
features = list(train)

for feature in features:
    print("Feature: {}".format(feature))
    uniques = train[feature].value_counts() #.unique()
    print("Amount: {}".format(uniques))


print("Done")

# correlation
"""
import seaborn as sns
import matplotlib.pyplot as plt
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

orig_train = orig_train.apply(sigmoid) # IMPORTANT!

new_ds = np.tanh(orig_train.values) # turns into numpoy array

f, ax = plt.subplots(figsize=(10, 8))
corr = orig_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.show()


f1, ax1 = plt.subplots(figsize=(10, 8))
corr1 = train.corr()
sns.heatmap(corr1, mask=np.zeros_like(corr1, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.show()

#test_coors = orig_train.drop("target", axis=1).apply(lambda x: x.corr(orig_train.target))

corr_matrix = orig_train.corr()
print(corr_matrix["target"].sort_values(ascending=False))

# VARS 25, 12, 15, 108 are higly correlated with each other
# Now find out which of these features lead to 0 or 1 in target
sorted_train_25 = orig_train.sort_values(by='var_25')
sorted_train_all = orig_train.sort_values(by=['var_25', 'var_12', 'var_108', 'var_15'])

print("Done")
"""
# %% Feature Tools
"""
test_ids = test['ID_code'] 
train_ids = train['ID_code']

transactions = train['target']


train_ids = train.index


y_df = np.array(df['target'])                        
df_ids = np.array(df.index)                     
df.drop(['ID_code', 'target'], axis=1, inplace=True)



combi = train.append(test, ignore_index=True)

train.drop(['ID_code', 'target'], axis=1, inplace=True)
test.drop(['ID_code'], axis=1, inplace=True)

combi.drop(['target'], axis=1, inplace=True)

print("Done")




# creating and entity set 'es'
es = ft.EntitySet(id = 'transactions')

# adding a dataframe 
es.entity_from_dataframe(entity_id = 'santander', dataframe = combi, index = 'ID_code')


print(es)
feature_matrix, feature_names = ft.dfs(entityset=es, 
target_entity = 'santander', 
trans_primitives = ['add_numeric', 'multiply_numeric'])

print("FM columns: " + feature_matrix.columns)

print(feature_matrix.head())

feature_matrix.drop(['Unnamed: 0'], axis=1, inplace=True)

feature_matrix = feature_matrix.reindex(index=combi['ID_code'])
feature_matrix = feature_matrix.reset_index()

print("Feature engineering done")

print(feature_matrix.columns)


"""


