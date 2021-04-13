#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:52:03 2020

@author: Edson Cilos
"""
#Standard modules
import os
import pandas as pd
from itertools import product


#Sklearn Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#Project modules
from utils import file_name as f_name
from param_grid import neural_grid, classical_grid
from pipeline import build_pipe


seed = 0

def search(scaler = '', 
           sav_filter = False,
           pca = True, 
           over_sample = True, 
           param_grid = classical_grid(), 
           prefix = '', 
           n_jobs = -2):
    
    
    print('Loading training set...')
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel() 
    
    
    print('Building pipeline...')
    pipe, file_name = build_pipe(scaler = scaler, 
                                 sav_filter = sav_filter,
                                 pca = pca, 
                                 over_sample = over_sample)
    
    file_name = prefix + file_name + 'gs.csv'
    
    print('The file name is: ' + file_name)
      
    cv_fixed_seed = StratifiedKFold(n_splits=4, 
                                    shuffle = True, 
                                    random_state = seed)

    print('Running parameter search (It can take a long time) ...')
    search = GridSearchCV(pipe,
                          param_grid,
                          scoring = 'neg_log_loss',
                          cv = cv_fixed_seed,
                          n_jobs = n_jobs,
                          verbose = 100)

    search = search.fit(X_train, y_train)

    results = pd.concat([pd.DataFrame(search.cv_results_["params"]),
                     pd.DataFrame(search.cv_results_['std_test_score'], 
                                  columns=["std"]),
                     pd.DataFrame(search.cv_results_["mean_test_score"], 
                                  columns=["neg_log_loss"])],axis=1)
                     
    results.sort_values(by=['neg_log_loss'], ascending=False, inplace=True)
            
    folder = os.path.join(os.getcwd(), 'results')
    
    if not os.path.exists(folder):
            os.makedirs(folder)
            
    final_path = os.path.join(folder, file_name)        
    print('Search is finished, saving results in ' + final_path)
    results.to_csv(final_path, index = False)
    
    return results

def run():
    
    i = 0
    
    for sv_filter, scaler, pca, over, nn in product([False, True], repeat = 5):
        
        i += 1
        
        grid = neural_grid() if nn else classical_grid()
        
        file_name = f_name(nn=nn,
                           sv_filter=sv_filter, 
                           scaler=scaler, 
                           pca= pca, 
                           over_sample= over)
        
        file_path = os.path.join(os.getcwd(), 'results', file_name)
        
        if os.path.isfile(file_path):
            print(file_name + " already exists, iteration was skipped ...")
            
        else:
            print("{0} iteration ({1}/32)...".format(file_name, str(i)))
            search(scaler = 'std' if scaler else '', 
                   sav_filter = sv_filter,
                   pca = pca, 
                   over_sample = over, 
                   param_grid = grid, 
                   prefix = 'nn_' if nn else '',
                   n_jobs = 1)
            
    print("GridSearch finished...")
            