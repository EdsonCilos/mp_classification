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
from sklearn.semi_supervised import SelfTrainingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#Project modules
from utils import build_row, load_encoder
import graphics

#Fix the seed of the evaluation
seed = 0

#Selected model: SelfTraining + SVC + pca + oversample
pre_model = SVC(kernel='linear', C = 10, probability = True, random_state = 0)
model = SelfTrainingClassifier(pre_model, threshold = 0.9,verbose=True)                               
scaler = False
pc = True
over_sample = True
                        
def final_test():
    
    
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
    #unlabeled data
    X_unlabel = pd.read_csv(os.path.join('data', 'unlabeled_data.csv')) 
    X_unlabel.drop(['3997.91411'], axis = 1, inplace=True)
    y_unlabel = -1*np.ones(X_unlabel.shape[0])
    
    
    X_test = pd.read_csv(os.path.join('data', 'X_test.csv')) 
    y_test = pd.read_csv(os.path.join('data', 'y_test.csv')).values.ravel()
    

    if(scaler):
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_unlabel = std.transform(X_unlabel)
        X_test = std.transform(X_test)
        
            
    if(pc):
        pca = PCA(n_components=0.99, random_state = seed)
        X_train = pca.fit_transform(X_train)
        X_unlabel = pca.transform(X_unlabel)
        X_test = pca.transform(X_test)
        
            
    if(over_sample):
        ros = RandomOverSampler(random_state = seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    
    X_semi = pd.concat([pd.DataFrame(X_train),
                            pd.DataFrame(X_unlabel)],
                           axis=0,
                           ignore_index=True)
    
    y_semi = np.concatenate((y_train, y_unlabel))
                
    model.fit(X_semi, y_semi)

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
    
        pre_model.fit(X_train, y_train)
        
        rows.append(build_row(X_test, y_test, pre_model.predict(X_test)))
     
        total_score.append([log_loss(y_test, pre_model.predict_proba(X_test)),
                     accuracy_score(np.array(y_test),pre_model.predict(X_test))
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
    