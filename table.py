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

def best_results():
    
    classical_models = {'SVC' : 'SVC', 
              'RandomForestClassifier' : 'RF', 
              'LogisticRegression' : 'LR', 
              'KNeighborsClassifier' : 'KNN', 
              'DecisionTreeClassifier' : 'DT',
              'GaussianNB' : 'GB'}
    
    #fig, axs = plt.subplots(1, 1, figsize=(16, 5))
    models = {}
    
    for sv_filter, scaler, pca, over, nn in product([False, True], repeat = 5):
        
        file_name = f_name(nn=nn,
                           sv_filter=sv_filter, 
                           scaler=scaler, 
                           pca= pca, 
                           over_sample= over)
        
        file_path = os.path.join(os.getcwd(), 'results', file_name)
        
        if os.path.isfile(file_path):
            
            df = pd.read_csv(file_path)
            replace = True
            
            if nn:
                
                row = df.iloc[0]
                
                if 'NN' in models:
                    if models['NN'][4] <= -row["neg_log_loss"]: 
                        replace=False
                            
                if replace:
                    models['NN'] = [int(sv_filter), int(scaler), int(pca),
                                    int(over), -row["neg_log_loss"], 
                                    row["std"]]
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
                        models[classical_models[key]] = [int(sv_filter), 
                                                         int(scaler),
                                                         int(pca),
                                                         int(over),
                                                         -row["neg_log_loss"], 
                                                         row["std"]]
        else:
            print(file_name + " does not exists, please run the gridSearch!")
            break
        
    data = []
    idxs = []
    
    for key in models:
        data.append(models[key])
        idxs.append(key)
        
    return pd.DataFrame(data = data, 
                        columns = ["Savitzky–Golay filter", 
                                   "Standard scaler",
                                   "PCA (99%)",
                                   "Over sample",
                                   "Log-loss", 
                                   "Standard Deviation"],
                            index = idxs).sort_values(by = ["Log-loss"])
    