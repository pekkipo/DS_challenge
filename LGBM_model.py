# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:59:57 2019

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
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import feature_engineering as fe
import get_models


gc.enable()


NAME = 'lgbm2'
    
def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path):
    
    model = get_models.get_lgbm_2()
    
    
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=3500, 
              early_stopping_rounds=3500)
      

    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, NAME, counter+1)
    model.booster_.save_model(save_to)
    
    return cv_val
    
def train_stage(df, lgb_path):
    
    
    y_df = np.array(df['target'])                        
    df_ids = np.array(df.index)                     
    #df.drop(['ID_code', 'target', 'Unnamed: 0'], axis=1, inplace=True)
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
        

    lgb_cv_result = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]
        #X_fit, y_fit = df[ids[0]], y_df[ids[0]]
        #X_val, y_val = df[ids[1]], y_df[ids[1]]
        
        # Added augemntation
        #X_fit, y_fit = fe.augment(X_fit, y_fit)
    
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path)
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)
    print('\nLightGBM VAL AUC: {}'.format(auc_lgb))
    return 0
    
    
def prediction_stage(df, lgb_path, submit=True):

    
    df.drop(['ID_code'], axis=1, inplace=True)
    
    lgb_models = sorted(os.listdir(lgb_path))
    lgb_result = np.zeros(df.shape[0])

    print('\nMake predictions...\n')
    
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)

    lgb_result /= len(lgb_models)
    
    if submit:
        submission = pd.read_csv('data/sample_submission.csv')
        submission['target'] = lgb_result
        submission.to_csv('lgb_starter_submission_01_04_FE.csv', index=False)


    return 0
    

############ RUN

train_path = 'data/train.csv'
test_path  = 'data/test.csv'
lgb_path = 'lgbm2_models/'

print('Load Train Data.')
df_train = pd.read_csv(train_path)
print('\nShape of Train Data: {}'.format(df_train.shape))

print('Load Test Data.')
df_test = pd.read_csv(test_path)
print('\nShape of Test Data: {}'.format(df_test.shape))



# Add features
df_train = fe.do_feature_engineering(df_train, 'train')
df_test = fe.do_feature_engineering(df_test, 'test')

#Create dir for models
#os.mkdir(lgb_path)

print('Train Stage.\n')
train_stage(df_train, lgb_path)

print('Prediction Stage.\n')
prediction_stage(df_test, lgb_path, True)

print('\nDone.')