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
from pipeline import pipe_config

classical_models = {'SVC' : 'SVC', 
              'RandomForestClassifier' : 'RF', 
              'LogisticRegression' : 'LR', 
              'KNeighborsClassifier' : 'KNN', 
              'DecisionTreeClassifier' : 'DT',
              'GaussianNB' : 'GB'}

def filter_results():
    
    df, estimator_path = best_results()
    
    log_loss = []
    models = []
    
    for model in estimator_path:
        
        _,scaler,pca,ov = pipe_config(os.path.basename(estimator_path[model]))
        
        file_name = f_name(nn = True if model == 'NN' else False, 
                           sv_filter = False, 
                           scaler = scaler, 
                           pca = pca, 
                           over_sample = ov)
                
        ft_file_name = 'ft_{}_'.format(model)  + f_name(nn = False,
                                                        sv_filter = True, 
                                                        scaler = scaler, 
                                                        pca = pca, 
                                                        over_sample = ov)
        

        file_path = os.path.join('results', file_name)
        ft_file_path = os.path.join('results', ft_file_name)
        
        df = pd.read_csv(file_path)
        ft_df = pd.read_csv(ft_file_path)
        
        row = df.iloc[0] if model == 'NN' else next(
                r  for _, r  in df.iterrows() if model in r['estimator'])
    
        ft_row = ft_df.iloc[0]
        
        log_loss.append([-row["neg_log_loss"], -ft_row["neg_log_loss"]])
        
        model_name = 'NN' if model == 'NN' else classical_models[model]
        models.append(model_name)
            
    result = pd.DataFrame(data = log_loss, 
                          columns = ["No filter", 
                                     "Best Savitzky–Golay filter"],
                          index = models)
    
    result.sort_values(by = ["No filter"], inplace = True)
        
    return result
            
    
def best_results():
    
    estimator_path = {}

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
                        models[classical_models[key]] = [int(sv_filter), 
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
                        columns = ["Savitzky–Golay filter", 
                                   "Standard scaler",
                                   "PCA (99%)",
                                   "Over sample",
                                   "Log-loss", 
                                   "Standard Deviation"],
                            index = idxs)
    
    df.sort_values(by = ["Log-loss"], inplace = True)
    
    return df, estimator_path
    