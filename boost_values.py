# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:13:48 2019

@author: Q466091
"""

import pandas as pd


sub = pd.read_csv("lgb_starter_submission_01_04_FE.csv")

"""
submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = lgb_result
submission.to_csv('lgb_starter_submission_01_04_FE.csv', index=False)
"""

targets = sub['target']


for index, row in sub.iterrows():
    
    obs_value = sub.at[index, 'target']
    if obs_value > 0.8:
        new_value = obs_value * 1.1
        if new_value < 1.0:
            sub.at[index, 'target'] = new_value
            print("Boosted from: {} to {} at index {}".format(obs_value, new_value, index))
    elif obs_value < 0.15:
        new_value = obs_value * 0.4
        if new_value > 0.0:
            sub.at[index, 'target'] = new_value
            print("Reduced from: {} to {} at index {}".format(obs_value, new_value, index))
            
sub.to_csv('lgb_starter_submission_01_04_FE_changed_2.csv', index=False)
