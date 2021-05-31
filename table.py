# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:18:35 2021

@author: Edson Cilos
"""
#Standard packages 
import os
import pandas as pd
from itertools import product

#Project packages
from utils import file_name as f_name

import config

classical_models = {'SVC' : 'SVC', 
              'RandomForestClassifier' : 'RF', 
              'LogisticRegression' : 'LR', 
              'KNeighborsClassifier' : 'KNN', 
              'DecisionTreeClassifier' : 'DT',
              'GaussianNB' : 'GB'}

    
def best_results():
    
    estimator_path = {}

    models = {}
    
    for scaler, baseline, pca, over, nn in product([False, True], repeat = 5):
        
        file_name = f_name(nn=nn,
                           baseline = baseline, 
                           scaler=scaler, 
                           pca= pca, 
                           over_sample= over)
        
        file_path = os.path.join(config._get_path('grid_search'), file_name)
        
        if os.path.isfile(file_path):
            
            df = pd.read_csv(file_path)
            replace = True
            
            if nn:
                
                row = df.iloc[0]
                
                if 'NN' in models:
                    if models['NN'][4] <= -row["neg_log_loss"]: 
                        replace=False
                            
                if replace:
                    
                    models['NN'] = [int(baseline),
                                    int(scaler), 
                                    int(pca),
                                    int(over), 
                                    -row["neg_log_loss"], 
                                    row["std"]]
                    
                    estimator_path['NN'] = file_path
            else:

                for key in classical_models:
        
                    row = next(r  for _, r  in df.iterrows() 
                               if key in r["estimator"])
                    
                    replace = True
                    
                    if classical_models[key] in models:
                        if (models[classical_models[key]][4]
                            <= -row["neg_log_loss"]): 
                            replace=False           
                            
                    if replace:
                        models[classical_models[key]] = [int(baseline),
                                                         int(scaler),
                                                         int(pca),
                                                         int(over),
                                                         -row["neg_log_loss"], 
                                                         row["std"]]
                        estimator_path[key] = file_path
        else:
            print(file_name + " does not exists, please run the gridSearch!")
            break
        
    data = []
    idxs = []
    
    for key in models:
        data.append(models[key])
        idxs.append(key)
        
    df = pd.DataFrame(data = data, 
                        columns = ["Baseline",
                                   "Standard scaler",
                                   "PCA (99%)",
                                   "Over sample",
                                   "Log-loss", 
                                   "Standard Deviation"],
                            index = idxs)
    
    df.sort_values(by = ["Log-loss"], inplace = True)
    
    return df, estimator_path
    