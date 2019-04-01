# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:12:44 2019

@author: Q466091
"""


import numpy as np
import pandas as pd
import data_specs as ds
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def form_dataset_for_one_var(dataset, mean_values, var):
    
    #targets = dataset['target']  
    #var_values = dataset[var]
    #ids = df.index  
    
    data_for_training = pd.DataFrame(columns=[
            'ID_code', 
            'value', 
            'diff_mean_0', 
            'diff_mean_1',
            'diff_median_0',
            'diff_median_1',
            'target'])
    
    
    def get_mean_and_median_values(var):
        var_row = mean_values.loc[mean_values['col'] == var]
        return var_row
    
        
    means_for_this_var = get_mean_and_median_values(var)
    
    counter = 0
    for index, row in dataset.iterrows():
        
        obs_id = dataset.at[index, 'ID_code']
        obs_value = dataset.at[index, var]
        diff_mean_0 = abs(abs(obs_value) - abs(means_for_this_var['mean_0'].values[0]))
        diff_mean_1 = abs(abs(obs_value) - abs(means_for_this_var['mean_1'].values[0]))
        diff_median_0 = abs(abs(obs_value) - abs(means_for_this_var['median_0'].values[0]))
        diff_median_1 = abs(abs(obs_value) - abs(means_for_this_var['median_1'].values[0]))
        target = dataset.at[index, 'target']
        
        row =  [obs_id, obs_value, diff_mean_0, diff_mean_1, diff_median_0, diff_median_1, target]
        
        data_for_training.loc[counter] = row
        
        counter += 1
        
    return data_for_training


def form_dataset_mixing(dataset, mean_values, features, var):
    
    features.remove(var)    
    
    columns=[
            'ID_code', 
            'target',
            'value', 
            'diff_mean_0', 
            'diff_mean_1',
            'diff_median_0',
            'diff_median_1'
            ]
    
    def format_col_name(x, var, sign):
        return x + '{}{}'.format(sign, var)
    
    features_plus = [format_col_name(x, var, '+') for x in features]
    features_minus = [format_col_name(x, var, '-') for x in features]
    columns.extend(features_plus)
    columns.extend(features_minus)
    
    data_for_training = pd.DataFrame(columns=columns)
    
    def get_mean_and_median_values(var):
        var_row = mean_values.loc[mean_values['col'] == var]
        return var_row
    
        
    means_for_this_var = get_mean_and_median_values(var)
    
    counter = 0
    for index, row in dataset.iterrows():
        
        permutations_with_the_feature_plus = []
        permutations_with_the_feature_minus = []
        
        obs_id = dataset.at[index, 'ID_code']
        target = dataset.at[index, 'target']
        obs_value = dataset.at[index, var]
        
        for feature in features:

            feature_value = dataset.at[index, feature]
            
            obs_plus = abs(obs_value) + abs(feature_value)
            obs_minus = abs(obs_value) - abs(feature_value)
            
            permutations_with_the_feature_plus.append(obs_plus)
            permutations_with_the_feature_minus.append(obs_minus)
            

        diff_mean_0 = abs(abs(obs_value) - abs(means_for_this_var['mean_0'].values[0]))
        diff_mean_1 = abs(abs(obs_value) - abs(means_for_this_var['mean_1'].values[0]))
        diff_median_0 = abs(abs(obs_value) - abs(means_for_this_var['median_0'].values[0]))
        diff_median_1 = abs(abs(obs_value) - abs(means_for_this_var['median_1'].values[0]))
        
        
        row =  [
                 obs_id, 
                 target,
                 obs_value,
                 diff_mean_0, 
                 diff_mean_1,
                 diff_median_0, 
                 diff_median_1          
                ]
        
        row.extend(permutations_with_the_feature_plus)
        row.extend(permutations_with_the_feature_minus)
        
        data_for_training.loc[counter] = row
        
        counter += 1
        
    return data_for_training

# Train algorithm
#targets = var_0_dataset['target']
#var_0_dataset.drop(['ID_code', 'target'], axis=1, inplace=True)

def fit_knn(X_fit, y_fit, X_test, path, name):
    
    bootstrap = True 
    max_depth = 100 
    max_features = 2 
    min_samples_leaf = 4 
    min_samples_split = 10 
    n_estimators = 1000 
    verbose = 0

    rf = RandomForestClassifier(n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            max_features=max_features, 
                            min_samples_leaf=min_samples_leaf, 
                            min_samples_split=min_samples_split, 
                            bootstrap=bootstrap,
                            verbose=verbose)
      
    rf.fit(X_fit, y_fit)
    cv_val = rf.predict_proba(X_test)[:,1]
    
    #Save knn Model
    save_to = '{}RF_{}.joblib'.format(path, name)
    dump(rf, save_to) 
    
    return cv_val

def fit_lgb(X_fit, y_fit, X_val, y_val, path, name):
    
    model = lgb.LGBMClassifier(objective = "binary", 
    boosting = "gbdt",
    metric="auc",
    boost_from_average=False,
    num_threads=8,
    learning_rate =0.0081,
    num_leaves =13,
    max_depth=-1,
    feature_fraction =0.041,
    bagging_freq =5,
    bagging_fraction =0.331,
    min_data_in_leaf =80,
    min_sum_hessian_in_leaf =10.0,
    verbosity =1,
    num_iterations =99999999,
    seed=44000,
    random_state=42)
    
    
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=500, 
              early_stopping_rounds=500)
      

    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save LightGBM Model
    save_to = '{}LGB_{}.txt'.format(path, name)
    model.booster_.save_model(save_to)
    
    return cv_val

def train(df, path, var):
        
    y_df = df['target'].astype('int')                           
    df = df.drop(['ID_code', 'target'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(df.values, y_df, test_size=0.33, random_state=42)
    
    #results = fit_knn(X_train, y_train, X_test, path, name=var)
    results = fit_lgb(X_train, y_train, X_test, y_test, path, name=var)

    auc_knn  = round(roc_auc_score(y_test, results), 4)
    print('\nKNN VAL AUC FOR VAR {} : {}'.format(var, auc_knn))
    
    return 0

def predict(df, path, var, mod = 'LGB'):
 
    df.drop(['ID_code', 'target'], axis=1, inplace=True)

    result = np.zeros(df.shape[0])

    print('\nMake predictions for var {}...\n'.format(var))
    
    # mod either LGB or RF
    
    """ IF SCIKIT MODEL like RF for instance
    model = load('{}{}_{}.joblib'.format(path, mod, var)) 
    knn_result = model.predict(df.values)
    """
    
    model = lgb.Booster(model_file='{}{}_{}.txt'.format(path, mod, var))
    result = model.predict(df.values)
    
    return result

def run_feature_creation(var, folder, features, load_file=False):
    
    # Form dataset comprising mean and median value differences for that feature - var
    training_datasets_path = '{}datasets/'.format(folder)
    if not load_file:
        #var_dataset = form_dataset_for_one_var(df_train, mean_and_median_values, var)
        var_dataset = form_dataset_mixing(df_train, mean_and_median_values, features, var)
        var_dataset.to_pickle("{}MIX_dataset_{}_cut.pkl".format(training_datasets_path, var))
        print('\n Dataset for {} is formed'.format(var))
    else:
        var_dataset = pd.read_pickle("{}MIX_dataset_{}_cut.pkl".format(training_datasets_path, var))
    
    # Train kNN
    predicted_path = '{}predicted_features/Mixed'.format(folder)
    train(var_dataset, folder, var)
    
    # Predict
    predicted_feature = predict(var_dataset, folder, var)
    predicted_feature = pd.DataFrame(predicted_feature)
    predicted_feature.to_pickle("{}predicted_feature_{}.pkl".format(predicted_path, var))
    print('\n Predicted feature for var {} was written into the file')


################### RUN

train_path = 'data/train_cut.csv'


print('Load Train Data.')
df_train = pd.read_csv(train_path, index_col=0)

mean_and_median_values = pd.read_pickle("mean_and_median_values_dataset.pkl")


# train one network for each feature, so 200 networks and for each row there will be 200 additional features
# actually might even think of using that as a replacement dataset

high_difference_features = ds.high_difference_features
moderate_difference_features = ds.moderate_difference_features

all_features = list(df_train)
all_features.remove('ID_code')
all_features.remove('target')

folder = 'RandomForestsMix/'

#for var in high_difference_features:
#    run_feature_creation(var, folder, all_features, False)
    
    
 #try to do automatic feature engineering
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


#feature_matrix = automatic_FE(df_train)
#feature_matrix.to_pickle("feature_matrix.pkl")
 
    


