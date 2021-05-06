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
from timeit import default_timer as timer

#Sklearn Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#Project modules
from utils import file_name as f_name
from param_grid import neural_grid, classical_grid, filter_grid
from pipeline import build_pipe, pipe_config
from table import best_results
from model_picker import get_1estimator

#Config module
import config

seed = config._seed()

def search(scaler = '', 
           sav_filter = False,
           pca = True, 
           over_sample = True, 
           param_grid = classical_grid(), 
           prefix = '', 
           n_jobs = 1,
           save = True):
    
    
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
      
    cv_fixed_seed = StratifiedKFold(n_splits = 5, shuffle = False)

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
    
    if save:
        
        folder = os.path.join('results')
    
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        final_path = os.path.join(folder, file_name)        
        print('Search is finished, saving results in ' + final_path)
        results.to_csv(final_path, index = False)
    
    return results

def run_gs():
    
    i = 0
    
    print('Steps: (1) GridSearch across several combinations \n \
          (2) GridSearch searching for best filters')
          
    for scaler, pca, over, nn in product([False, True], repeat = 4):
        
        i += 1
        
        grid = neural_grid() if nn else classical_grid()
        
        file_name = f_name(nn=nn,
                           sv_filter=False, 
                           scaler=scaler, 
                           pca= pca, 
                           over_sample= over)
        
        folder = os.path.join('results')
    
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        file_path = os.path.join(folder, file_name)
        
        if os.path.isfile(file_path):
            print(file_name + " already exists, iteration was skipped ...")
            
        else:
            print("{0} iteration ({1}/16)...".format(file_name, str(i)))
            start = timer()
            search(scaler = 'std' if scaler else '', 
                   sav_filter = False,
                   pca = pca, 
                   over_sample = over, 
                   param_grid = grid, 
                   prefix = 'nn_' if nn else '',
                   n_jobs = 1)
            end = timer()
            append_time(file_name, str(end - start))
            
    print("GridSearch finished...")
    print("Building best results table...")
    
    df, paths = best_results()
    
    print("Trying diferent filters for each algorithm...")
    j = 0
    
    for model in paths:
        
        j +=1
        file_path = paths[model]
        
        estimator, config = get_1estimator(file_path = file_path, 
                                           model_name = model)
                                   
        print('{} best configuration: {}'.format(model, config))
        
        param_grid = filter_grid(estimator)
        
        _, scaler, pca, over_sample = pipe_config(os.path.basename(
            file_path))
        
        prefix = 'ft_{}_'.format(model)
        
        file_name = prefix  + f_name(nn = False,
                                     sv_filter = False,
                                     scaler = scaler, 
                                     pca = pca, 
                                     over_sample = over_sample)
        
        file_path = os.path.join('results', file_name)
        
        if os.path.isfile(file_path):
            print("{} already exists, filter iteration ({}/7) was skipped ..."
                  .format(file_name, j))
                  
        else:
            
            print("{} filter iteration ({}/7)...".format(file_name, j))
            start = timer()
            results = search(scaler = scaler, 
                             sav_filter = True,
                             pca = pca, 
                             over_sample = over_sample, 
                             param_grid = param_grid, 
                             prefix = prefix, 
                             n_jobs = 1,
                             save = False)            
            print('Search is finished, saving results in ' + file_path)
            results.to_csv(file_path, index = False)
            
            end = timer()
            
            append_time(file_name, str(end - start))
    
    print('GridSearch fully finished.')
    
    
def append_time(file_name, time):
    with open(os.path.join('results', "time.csv"), "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        file_object.write("{},{}".format(file_name, time))

        