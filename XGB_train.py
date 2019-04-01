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
import joblib
import xgboost as xgb


gc.enable()


NAME = 'xgb_1'
    
def fit_lgb(d_train, d_val, X_val, y_val, counter, lgb_path):
    
    #model = get_models.get_xgboost()
    
    params = {'tree_method': 'hist',
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'learning_rate': 0.0936165921314771,
                 'max_depth': 2,
                 'colsample_bytree': 0.3561271102144279,
                 'subsample': 0.8246604621518232,
                 'min_child_weight': 53,
                 'gamma': 9.943467991283027,
                 'silent': 1}
    
    model = xgb.train(params=params, dtrain=d_train, num_boost_round=4000, evals=[(d_train, "Train"), (d_val, "Val")],
        verbose_eval= 100, early_stopping_rounds=50) 
    
    cv_val = model.predict(xgb.DMatrix(X_val))
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.joblib'.format(lgb_path, NAME, counter+1)
    #save model
    joblib.dump(model, save_to) 

    return cv_val
    
def train_stage(df, lgb_path):
    
    
    y_df = df['target']   
    train_cols = [c for c in df.columns if c not in ["ID_code", "target"]]                   
    df_ids = np.array(df.index)                     
    #df.drop(['ID_code', 'target', 'Unnamed: 0'], axis=1, inplace=True)
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    


    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    
    print('\nModel Fitting...')
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(df, y_df)):
    
        X_fit, y_fit = df[train_cols].iloc[trn_idx], y_df.iloc[trn_idx]
        X_val, y_val = df[train_cols].iloc[val_idx], y_df.iloc[val_idx]
        
        # Added augemntation
        #X_fit, y_fit = fe.augment(X_fit, y_fit)
        
        d_train = xgb.DMatrix(X_fit, y_fit, feature_names=X_fit.columns)
        d_val = xgb.DMatrix(X_val, y_val, feature_names=X_val.columns)
        
        oof_preds[val_idx] = clf.predict(xgb.DMatrix(val_x))
        sub_preds += clf.predict(xgb.DMatrix(test_df[train_cols])) / folds.n_splits
    
        lgb_cv_result[ids[1]] += fit_lgb(d_train, d_val, X_val, y_val, counter, lgb_path)
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
        #model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        model = joblib.load('{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)

    lgb_result /= len(lgb_models)
    
    if submit:
        submission = pd.read_csv('data/sample_submission.csv')
        submission['target'] = lgb_result
        submission.to_csv('lgb_starter_submission_01_04_FE.csv', index=False)


    return 0
    

############ RUN

train_path = 'data/train_cut.csv'
test_path  = 'data/test_cut.csv'
lgb_path = 'xgb1_models/'

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