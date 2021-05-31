#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:26:35 2020

@author: edson
"""
#os, sklearn, imbalanced learn, pandas and numpy
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.svm import SVC

#Project modules
from utils import build_row, load_encoder
from baseline import als
import graphics
import config
from param_grid import build_nn


#Tensoflow
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

#Fix the seed of the evaluation
seed = config._seed()

#Selected model: SVC + pca + oversample
base_model = SVC(kernel = 'linear', C = 100, probability = True, 
                 random_state= 0)
            
baseline= True
scaler = False
pc = False
over_sample = True
                        
def final_test():
    
    model = clone(base_model)
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
   
        
    X_test = pd.read_csv(os.path.join('data', 'X_test.csv')) 
    y_test = pd.read_csv(os.path.join('data', 'y_test.csv')).values.ravel()
    
    if(baseline):
        print('Applying baseline correction...')
        for idx, row in X_train.iterrows():
            X_train.iloc[idx, :] = row - als(row)
        for idx, row in X_test.iterrows():
            X_test.iloc[idx, :] = row - als(row)
            

    if(scaler):
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
            
    if(pc):
        pca = PCA(n_components=0.99, random_state = seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
            
    if(over_sample):
        ros = RandomOverSampler(random_state = seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    
                
    model.fit(X_train, y_train)

    total_scores = [log_loss(y_test, model.predict_proba(X_test)),
                    accuracy_score(np.array(y_test), model.predict(X_test))
                    ]
    
    return _results(
            [build_row(X_test, y_test, model.predict(X_test))], #detailed score
            [total_scores], 
            'final_test', 
            load_encoder()), total_scores
    
                        
#we shall use only the pre_model
def mccv_all_data():
    
    #Load and prepare the dataset
    dataset = pd.read_csv(os.path.join('data','D4_4_publication.csv'))
    dataset.drop(['Nom '], axis=1, inplace=True)
    dataset.rename(columns={'Interpretation ': "label"}, inplace=True)
    
    freq = dataset["label"].value_counts()
    less_rep = [idx for idx, value in freq.items()  if value < 5]
    
    for i, row in dataset.iterrows():
      if row["label"] in less_rep:
          dataset.at[i, 'label'] = 'Unknown'


    #Encoding the labels
    encoder = LabelEncoder()
    dataset["label"] = encoder.fit_transform(dataset["label"])

    #(X,y) is our labeled sample
    X = dataset.drop(["label"], axis = 1)
    y = dataset["label"].copy()
    
    rows = []
    
    total_score = []
                        
                        
    for j in range(1000):
        
        print("Iteration number: " + str(j) + "--------- final model")
        
        model = clone(base_model)
     
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       stratify=y, 
                                                       test_size= 1/3,
                                                       random_state = j)
        #d4_publication already scaled
            
        if(pc):
            pca = PCA(n_components=0.99, random_state = seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        
            
        if(over_sample):
            ros = RandomOverSampler(random_state = seed)
            X_train, y_train = ros.fit_resample(X_train, y_train)
    
        model.fit(X_train, y_train)
        
        rows.append(build_row(X_test, y_test, model.predict(X_test)))
     
        total_score.append([log_loss(y_test, model.predict_proba(X_test)),
                     accuracy_score(np.array(y_test), model.predict(X_test))
                     ])

    return _results(rows, total_score, 'final_model_mccv_all_data', encoder),\
            np.array(total_score).mean(axis=0)


def _results(detailed_score_rows, total_score, result_name, encoder):
    
    pd.DataFrame(data = [np.array(total_score).mean(axis=0)], 
                 columns= ['Cross_Entropy', 'Accuracy'])\
    .to_csv(os.path.join('results', result_name + '_total_score.csv'),
            index=False)
    
    df = pd.DataFrame(data = detailed_score_rows)
    
    df.to_csv(os.path.join('results', result_name + '_detailed_score.csv'),
              index=False)
    
    graphics.detailed_score_heatmap(df, result_name)
    
    return df
    