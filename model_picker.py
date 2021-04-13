# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:54:51 2021

@author: Edson Cilos
"""
#Standard Packages 
import pandas as pd

#Sklearn API
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def classical_1estimator(file_path, model_name = 'SVC'):
    
    estimator_dic = {'SVC' : SVC, 
              'RandomForestClassifier' : RandomForestClassifier, 
              'LogisticRegression' : LogisticRegression, 
              'KNeighborsClassifier' : KNeighborsClassifier, 
              'DecisionTreeClassifier' : DecisionTreeClassifier,
              'GaussianNB' : GaussianNB}
    
    if model_name not in estimator_dic:
        
        raise Exception("Model '{}' not available in the dictionary. Choose one \
                        of the following: {} {}".format(model_name, 
                        ', '.join([key for key in estimator_dic]),
                        '.'))
    
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    row = next(r  for _, r  in df.iterrows() if model_name in r['estimator'])
    

    txt = row['estimator'].split('(')[1].split(')')[0].replace(' ', '')
    
    hyper_params = {}
    
    for hyper_value in txt.split(','):
         x =  hyper_value.split('=')
         hyper_params[x[0]] = _parser(x[1])
            
    for col in [x for x in df.columns if 'estimator__' in x]:
        
        param_name = col.split('__')[1]		
        
        if row[col] != '': 
            hyper_params[param_name] = _parser(row[col])
            
    return estimator_dic[model_name](**hyper_params), hyper_params
            
def _parser(value):    
    
    try: 
        return float(value)
    except:
        if value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            return value
        
        
    
    
    