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
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

#Project modules
from utils import Remove_less_representative, Savgol_transformer, build_row, \
load_encoder

import graphics

#Fix the seed of the evaluation
seed = 0
remove = 4 #Fix the number of removed classes

model = Pipeline([
                        ('sg_filter', Savgol_transformer(11, 10)),
                        ('std_scaler', StandardScaler()),
                        ('over_sample', RandomOverSampler(random_state = seed)),
                        ('estimator', SVC(C = 0.002,
                                          kernel = 'linear',
                                          probability=True, 
                                          random_state = seed))
             ])
                        
def final_test():
    
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
        
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
    X_test = pd.read_csv(os.path.join('data', 'X_test.csv')) 
        
    y_test = pd.read_csv(os.path.join('data', 'y_test.csv')).values.ravel()
    
    
    model.fit(X_train, y_train)

    total_scores = [log_loss(y_test, model.predict_proba(X_test)),
                    accuracy_score(np.array(y_test), model.predict(X_test))
                    ]
    
    return _results(
            [build_row(X_test, y_test, model.predict(X_test))], #detailed score
            [total_scores], 
            'final_test', 
            load_encoder()), total_scores
    
                        

def mccv_all_data():
    
    #Load and prepare the dataset
    dataset = pd.read_csv(os.path.join('data','D4_4_publication.csv'))
    dataset.drop(['Nom '], axis=1, inplace=True)
    dataset.rename(columns={'Interpretation ': "label"}, inplace=True)


    df = Remove_less_representative(dataset, remove)


    #Encoding the labels
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    #(X,y) is our labeled sample
    X = df.drop(["label"], axis = 1)
    y = df["label"].copy()
    
    rows = []
    
    total_score = []
                        
                        
    for i in range(1000):
    
     print("Iteration number: " + str(i) + "--------- final model")
    
     X_train, X_val, y_train, y_val = train_test_split(X, 
                                                       y, 
                                                       stratify=y, 
                                                       test_size= 1/3,
                                                       random_state = seed)
     
     model.fit(X_train, y_train)
     
     rows.append(build_row(X_val, y_val, model.predict(X_val)))
     
     total_score.append([log_loss(y_val, model.predict_proba(X_val)),
                     accuracy_score(np.array(y_val), model.predict(X_val))
                     ])
     
    
    return _results(rows, total_score, 'final_model_mccv_all_data', encoder),\
            np.array(total_score).mean(axis=0)

def _results(detailed_score_rows, total_score, result_name, encoder):
    
    pd.DataFrame(data = [np.array(total_score).mean(axis=0)], 
                 columns= ['Cross_Entropy', 'Accuracy'])\
    .to_csv(os.path.join('results', result_name + '_total_score.csv'),
            index=False)
    
    df = pd.DataFrame(data = detailed_score_rows)
    
    df.to_csv(os.path.join('results', result_name + '_detailed_score.csv'))
    
    graphics.detailed_score_heatmap(df, result_name)
    
    return df
    