# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:59:48 2019

@author: Q466091
"""

import numpy as np

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
import numpy as np
import utils
import params
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
from sklearn.model_selection import KFold
import gc
import pandas as pd
import feature_engineering as fe



def get_nn_model2(input_dim, load = False):

    model = Sequential()
    model.add(Dense(250, input_shape=(input_dim,),  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(250,  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250,  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(250,  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    # last layer    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
                  metrics=['accuracy'],
                  optimizer='adam')
    
    if load:
        # Load pretrained weigths
        model.load_weights(params.weights_nn3)  
    
    return model


def get_nn_model(input_dim, load = False):

    model = Sequential()
    model.add(Dense(200, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150))
    model.add(Activation('relu'))
    # last layer    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
                  metrics=['accuracy'],
                  optimizer='adam')
    
    if load:
        # Load pretrained weigths
        model.load_weights(params.weights_nn3)  
    
    return model


## 
def train_with_folds(X_train, X_test, y_train):
    
    input_dim = X_train.shape[1]
    
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    sub_preds = np.zeros(X_test.shape[0])
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        trn_x, trn_y = X_train[trn_idx], y_train[trn_idx]
        val_x, val_y = X_train[val_idx], y_train[val_idx]
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.0000001, mode='min')
        
        h5_path = "nn_models/NN_model.h5"
        checkpoint = ModelCheckpoint(h5_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        
        print( 'Setting up neural network...' )
    
        nn = get_nn_model(input_dim)
        nn.compile(loss='binary_crossentropy', optimizer='adam')
        
        print( 'Fitting neural network...' )
        nn.fit(trn_x, trn_y, validation_data = (val_x, val_y), epochs=params.epochs, verbose=2,
              callbacks=[reduce_lr, checkpoint, early_stopping, utils.roc_auc_callback(training_data=(trn_x, trn_y),validation_data=(val_x, val_y))])
        nn.load_weights(h5_path)
        print( 'Predicting...' )
        sub_preds += nn.predict(X_test).flatten().clip(0,1) / folds.n_splits
        
        gc.collect()
        
        return sub_preds
        
        
def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x
        
def prepare_data(df_train, df_test):
    
    
    y_train = df_train['target'].values
    
    X_train = df_train.drop(columns=['ID_code', 'target'])
    
    X_test = df_test.drop(columns=['ID_code'])
    
    for i in X_train.columns:
        #print('Categorical: ',i)
        X_train[i] = rank_gauss(X_train[i].values)
    
    for i in X_test.columns:
        #print('Categorical: ',i)
        X_test[i] = rank_gauss(X_test[i].values)
        
    X_train = X_train.values
    X_test = X_test.values
    
    return X_train, X_test, y_train


def submit(df_test, sub_preds, name):
    sub = pd.DataFrame()
    sub["ID_code"] = df_test["ID_code"]
    sub["target"] = sub_preds
    sub.to_csv("submissions/{}.csv".format(name), index=False)
    print("Submission file is ready")
    
    
    
###### RUN
train_path = 'data/train.csv'
test_path  = 'data/test.csv'
print('Load Train Data.')
df_train = pd.read_csv(train_path)
print('\nShape of Train Data: {}'.format(df_train.shape))

print('Load Test Data.')
df_test = pd.read_csv(test_path)
print('\nShape of Test Data: {}'.format(df_test.shape))

# Add features
df_train = fe.do_feature_engineering(df_train)
df_test = fe.do_feature_engineering(df_test)
    
  
X_train, X_test, y_train = prepare_data(df_train, df_test)
sub_preds = train_with_folds(X_train, X_test, y_train)

submit(df_test, sub_preds, 'NN_cv_first_try')
