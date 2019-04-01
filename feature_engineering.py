# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:49:59 2019

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
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import re


#################### FUNCTIONS

def combine_81_12(df):
    
    df['var_12_81_1'] = df['var_81'] + df['var_12']       
        
    #df.drop(['var_12', 'var_81'], axis=1, inplace=True)
    return df

def combine_96_94(df):
    
    df['var_96_94_1'] = df['var_96'] + df['var_94']       
        
    #df.drop(['var_12', 'var_81'], axis=1, inplace=True)
    return df

def combine_13_9(df):
    
    df['var_13_9_1'] = df['var_13'] + df['var_9']       
        
    #df.drop(['var_12', 'var_81'], axis=1, inplace=True)
    return df

def combine_2_3(df):
    
    df['var_2_3_1'] = df['var_2'] - df['var_3']
        
    #df.drop(['var_12', 'var_81'], axis=1, inplace=True)
    return df

def combine_68_108(df):
    
    df['var_68_108_1'] = df['var_108'] + df['var_68']   
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df

def combine_108_109(df):
    
    df['var_108_109_1'] = df['var_109'] - df['var_108']   
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df

def combine_81_108(df):
    
    df['var_81_108_1'] = df['var_81'] - df['var_108']   
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df

def combine_25_15(df):
    
    df['var_25_15_1'] = df['var_25'].apply(np.sqrt) + df['var_15'].apply(np.sqrt)   
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df


def combine_126_68(df):
    
    df['var_126_68_1'] = df['var_126'] - df['var_68']  
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df

def combine_126_125(df):
    
    df['var_126_125_1'] = df['var_126'] - df['var_125']  
    
    #df.drop(['var_68', 'var_108'], axis=1, inplace=True)
    return df


def divide_x(x):
    return 1/x

def cube_root(x):
    return x ** (1/3)

def sq(x):
    return pow(x,2)

def apply_transforms(dataset):
    """
    dataset['var_108'] = dataset['var_108'].apply(np.log)
    dataset['var_81'] = dataset['var_81'].apply(np.log)
    dataset['var_12'] = dataset['var_12'].apply(np.log)
    """
    # Transform all columns with only positive values to log
    
    for column in dataset:
        if ((column != 'ID_code') and (column != 'target') and (column != 'Unnamed: 0') and (dataset[column] > 0).all()):
            dataset[column] = dataset[column].apply(np.log)
       # elif ((column != 'ID_code') and (column != 'target') and (column != 'Unnamed: 0')):
        #    dataset[column] = dataset[column].apply(sq)
        
    
    return dataset
    

def transform_selected(dataset):
    
    dataset['var_80'] = dataset['var_80'].apply(np.cbrt)
    
    
    return dataset


def drop_some(dataset):
    
    features_to_drop = [
                        'var_185',  
                        'var_158',
                        'var_30',
                        'var_38',
                        'var_17',
                        'var_27',
                        'var_41',
                        'var_124'
                       ]
    
    for feature in features_to_drop:
        dataset.drop([feature], axis=1, inplace=True)
        
    return dataset    

def bin_more(dataset):
    
    feats = [
            ('var_94', 5),
            ('var_6', 5),
            ('var_108', 6),
            ('var_12', 5),
            ('var_15', 5),
            ('var_81', 5),
            ('var_23', 5),
            ('var_25', 4),
            ('var_34', 5),
            ('var_42', 5)
             ]
    
    for feat in feats:
        bins = ["{}".format(num+1) for num in range(feat[1])]
        new_feat = pd.cut(dataset[feat[0]], feat[1], labels = bins)
        new_feat = new_feat.to_frame()
        new_feat = pd.get_dummies(new_feat)
        
        # Can remove original features or try to keep them
        #dataset.drop([feat[0]], axis=1, inplace=True)
        
        dataset = pd.concat([dataset, new_feat],axis = 1)
        
    return dataset
    
    
def automatic_FE(df):
    
    import featuretools as ft
    
    # Make an entityset and add the entity
    es = ft.EntitySet(id = 'santander')
    es.entity_from_dataframe(entity_id = 'transactions', dataframe = df, 
                             make_index = True, index = 'index')
    
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'transactions',
                                      trans_primitives = ['add_numeric', 'multiply_numeric'])
    
    feature_matrix.head()
    
    
    return feature_matrix

def augment(x,y,t=2):
    
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

def add_statistics_features_for_columns():
    
    # Statistics for the columns basically
    
    train_path = 'data/train.csv'
    test_path  = 'data/test.csv'
    df_train = pd.read_csv(train_path)
    print('\nShape of Train Data: {}'.format(df_train.shape))
    
    print('Load Test Data.')
    df_test = pd.read_csv(test_path)
    print('\nShape of Test Data: {}'.format(df_test.shape))
    
    df_train = df_train.drop(['ID_code','target'], axis=1)
    
    len_train = len(df_train)

    df_test = df_test.drop(['ID_code'], axis=1)
    
    merged = pd.concat([df_train, df_test])
      
    idx = merged.columns.values[0:200]
    for df in [merged]:
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    
    train_new_features = merged.iloc[:len_train, 200:208]
    test_new_features = merged.iloc[len_train:, 200:208]
    
    train_new_features.to_pickle("eight_add_train_dataset.pkl")
    test_new_features.to_pickle("eight_add_test_dataset.pkl")
   
    print("New features are ready and saved")
    
    return train_new_features, test_new_features


def add_statistics_features_for_rows(dataset):
    
    train_path = 'data/train.csv'
    test_path  = 'data/test.csv'
    df_train = pd.read_csv(train_path)
    print('\nShape of Train Data: {}'.format(df_train.shape))
    
    print('Load Test Data.')
    df_test = pd.read_csv(test_path)
    print('\nShape of Test Data: {}'.format(df_test.shape))
    
    df_train = df_train.drop(['ID_code','target'], axis=1)
    
    len_train = len(df_train)

    df_test = df_test.drop(['ID_code'], axis=1)
    
    merged = pd.concat([df_train, df_test], axis=0)
    
    def get_means_and_medians(dataset):
        
        means_df = pd.DataFrame()
        medians_df = pd.DataFrame()
        
        for var in dataset:
            mean_val = dataset.loc[:, var].mean()
            median_val = dataset.loc[:, var].median()
            means_df.at[0, var] = mean_val
            medians_df.at[0, var] = median_val
            
        means_df.to_pickle("mean_values_dataset.pkl")
        means_df.to_pickle("median_values_dataset.pkl")
            
        return means_df, medians_df
    
    
    
    def add_new_features_for_each_var(dataset, var, means_df, medians_df):
        
        new_features_dataframe = pd.DataFrame()
        
        for index, row in dataset.iterrows():
            
            obs_value = dataset.at[index, var]
            mean_val = means_df.at[0, var]
            median_val = medians_df.at[0, var]
            difference_var_mean = abs(obs_value - mean_val)
            difference_var_median = abs(obs_value - median_val)
                        
            col_mean = var + '_mean_diff'
            col_median = var + '_median_diff'              
            new_features_dataframe.at[index, col_mean] = difference_var_mean
            new_features_dataframe.at[index, col_median] = difference_var_median
            
            print("Index done: {}".format(index))
            
        print("Done")    
        return new_features_dataframe
            
        
    means, medians = get_means_and_medians(merged)
    
    del merged
    
    new_features_train_dataframe = pd.DataFrame()
    new_features_test_dataframe = pd.DataFrame()
    for var in df_train:
        one_var = add_new_features_for_each_var(df_train, var, means[:200000], medians[:200000])
        new_features_train_dataframe = pd.concat([new_features_train_dataframe, one_var], axis=1)
        new_features_train_dataframe.to_pickle("{}_train_dataset.pkl".format(var))
        
    for var in df_test:
        one_var = add_new_features_for_each_var(df_test, var, means[200000:], medians[200000:])
        new_features_test_dataframe = pd.concat([new_features_test_dataframe, one_var], axis=1)
        new_features_train_dataframe.to_pickle("{}_test_dataset.pkl".format(var))
        

    
    new_features_train_dataframe.to_pickle("train_enhanced_dataset.pkl")
    new_features_test_dataframe.to_pickle("test_enhanced_dataset.pkl")
    
    del df_train, df_test, means, medians
    gc.collect()
        
    return new_features_train_dataframe, new_features_test_dataframe
        
            

### HANDLING DATE
    
# OLD FUNC
def handle_var68_date(dataset):
    
    epoch_datetime = pd.datetime(1900, 1, 1)
    trf_var_68_s = (dataset['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
    date_s68 = trf_var_68_s.map(datetime.fromordinal)
    
    #dataset['date68'] = date_s68
    dataset['year68'] = date_s68.map(lambda x: x.year)
    dataset['month68'] = date_s68.map(lambda x: x.month)
    dataset['weekday68'] = date_s68.map(lambda x: x.weekday())
    dataset['day68'] = date_s68.map(lambda x: x.day)
    
    #dataset.drop(['date68'], axis=1, inplace = True) #.sort_values('date68')
    # can do it like this but don't forget inplace True!
    
    return dataset

def calculateDate(ordinal, _epoch0=datetime(1899, 12, 31)):
    ordinal = (ordinal*10000)-7000
    if ordinal > 59:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)


def add_datepart(df, fldname, drop=True, time=False, errors="raise"):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
    
    
def handle_date(dataset):
    
    dates = []
    for i in range(len(dataset)):
        dates.append(calculateDate(dataset["var_68"][i]))
    
    dataset["date"] = dates
    add_datepart(dataset, "date")
    
    # Convert to 0 and 1 int so that Im not annoyed by T/F
    dataset['Is_month_end'] = dataset['Is_month_end'].astype(int)
    dataset['Is_month_start'] = dataset['Is_month_start'].astype(int)
    dataset['Is_quarter_end'] = dataset['Is_quarter_end'].astype(int)
    dataset['Is_quarter_start'] = dataset['Is_quarter_start'].astype(int)
    dataset['Is_year_end'] = dataset['Is_year_end'].astype(int)
    dataset['Is_year_start'] = dataset['Is_year_start'].astype(int)

##################### MAIN FUNCTION
    

def do_feature_engineering_test(dataset):
    
    # First do binning of some features
    #dataset = bin_vars(dataset)
    
    # All below actually improves the score by 0.05
    """
    dataset = combine_81_12(dataset)
    dataset = combine_25_15(dataset)
    dataset = combine_96_94(dataset)
    dataset = combine_2_3(dataset)
    dataset = combine_108_109(dataset)
    dataset = combine_126_68(dataset)
    """
    #new_feats = automatic_FE(dataset)
    #dataset = drop_some(dataset)
   # dataset = handle_var68_date(dataset)
    
    print("FE is done")
    
    return dataset


def do_feature_engineering(dataset, mode='train'):
    
    # Statistic stuff has to go first, before adding other features
    #train_stats_feats, test_stats_feats = add_statistics_features_for_columns()
    # READ INSTEAD
    train_stats_feats = pd.read_pickle("eight_add_train_dataset.pkl")
    test_stats_feats = pd.read_pickle("eight_add_test_dataset.pkl")
    
    
    # takes too long, for now I will switch it off
    #train_stats_feats_2, test_stats_feats_2 = add_statistics_features_for_rows(dataset)
    
    if mode == 'train':
        dataset = pd.concat([dataset, train_stats_feats], axis=1)
    else:
        dataset = pd.concat([dataset, test_stats_feats], axis=1)
      

    dataset = combine_81_12(dataset)
    dataset = combine_25_15(dataset)
    dataset = combine_96_94(dataset)
    dataset = combine_2_3(dataset)
    dataset = combine_108_109(dataset)
    dataset = combine_126_68(dataset)
    
    handle_date(dataset)
    
    
    print("FE is done")
    
    return dataset