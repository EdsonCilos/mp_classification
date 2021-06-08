# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 03:51:25 2021

@author: scien
"""
import os
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC


import config


#Fix the seed of the evaluation
seed = config._seed()

def save_model():
    
    model = SVC(kernel = 'linear', C = 100, probability = True, 
                random_state= seed)
    
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    

    X_test = pd.read_csv(os.path.join('data', 'X_test.csv')) 
    y_test = pd.read_csv(os.path.join('data', 'y_test.csv')).values.ravel()
    
    X = pd.concat([X_train, X_test])
    y = np.concatenate((y_train, y_test))
    
    ros = RandomOverSampler(random_state = seed)
    X, y = ros.fit_resample(X, y)      
    model.fit(X, y)
    
    #save model
    pickle.dump(model, open(os.path.join("data", 'classifier.sav'), 'wb'))

def load_model():
    return pickle.load(open(os.path.join('data', 'classifier.sav'), 'rb'))

if __name__ == "__main__":
    
    if not os.path.exists(os.path.join("data", 'classifier.sav')):
        save_model()
        
    model = load_model()
