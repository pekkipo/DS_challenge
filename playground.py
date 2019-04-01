# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:00:15 2019

@author: Q466091
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import statistics


train_path = 'data/train_cut.csv'
test_path  = 'data/test_cut.csv'

print('Load Train Data.')
df_train = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)


y_df = np.array(df_train['target'])                        
df_ids = np.array(df_train.index)     
              
df_train.drop(['ID_code'], axis=1, inplace=True)

#X_train = df_train.values  


sums = df_train.sum(axis = 1, skipna = True) 
sums_divided = sums / 200 

print(sums)

sums_con = np.column_stack((y_df, sums, sums_divided))

# So check means for both target 0 and 1


"""
count_row = df.shape[0]  # gives number of row count
count_col = df.shape[1]  # gives number of col count
"""

def get_means(df, col):
    
    
    sum_0 = 0
    sum_1 = 0
    
    for index, row in df.iterrows():
        if (df_train.at[index, 'target'] == 0):
            sum_0 += df_train.at[index, col]
        else:
            sum_1 += df_train.at[index, col] # the same is df_train.loc[index, col]
            
    mean_0 = sum_0 / df.shape[0]
    mean_1 = sum_1 / df.shape[0]
    
    
            
    print("Var: {} - Mean_0: {}, Mean_1: {}, Sum_0: {}, Sum_1: {}".format(col, mean_0, mean_1, sum_0, sum_1))
    print("")
    
    return (col, mean_0, mean_1, sum_0, sum_1)

def get_medians(df, col):
    
    values_0 = []
    values_1 = []
    
    for index, row in df.iterrows():
        if (df_train.at[index, 'target'] == 0):
            values_0.append(df_train.at[index, col])
        else:
            values_1.append(df_train.at[index, col])
            
    median_0 = statistics.median(values_0)
    median_1 = statistics.median(values_1)
    
            
    print("Var: {} - Median_0: {}, Median_1: {}".format(col, median_0, median_1))
    print("")
    
    return (col, median_0, median_1)
    

#get_means(df_train, 'var_0')

def add_mean_data(df_train):
    
    data = pd.DataFrame(columns=['col', 'mean_0', 'mean_1', 'sum_0', 'sum_1'])
    counter = 0
    
    for column in df_train:
        
        col, mean_0, mean_1, sum_0, sum_1 = get_means(df_train, column)
        row =  [col, mean_0, mean_1, sum_0, sum_1]
        
        data.loc[counter] = row
        
        counter += 1
        
    return data



def add_median_data(df_train):
    
    data = pd.DataFrame(columns=['col', 'median_0', 'median_1'])
    counter = 0
    
    for column in df_train:
        
        col, median_0, median_1 = get_medians(df_train, column)
        row =  [col, median_0, median_1]
        
        data.loc[counter] = row
        
        counter += 1
        
    return data


def add_mean_and_median(df_train):
    
    data = pd.DataFrame(columns=['col', 'mean_0', 'mean_1', 'median_0', 'median_1'])
    counter = 0
    
    for column in df_train:
        
        col, mean_0, mean_1, sum_0, sum_1 = get_means(df_train, column)
        col, median_0, median_1 = get_medians(df_train, column)
        row =  [col, mean_0, mean_1, median_0, median_1]
        
        data.loc[counter] = row
        
        counter += 1
    
    
    return data
    
    
#data = add_mean_data(df_train)
#data.to_pickle("mean_values_dataset.pkl")
    

data = add_mean_and_median(df_train)
data.to_pickle("mean_and_median_values_dataset.pkl")

    
    
