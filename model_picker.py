# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:54:51 2021

@author: Edson Cilos
"""
#Standard Packages 
import pandas as pd
import numpy as np

#Sklearn API
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Tensorflow API
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

#Project modules
from param_grid import build_nn
from table import best_results

estimator_dic = {'SVC' : SVC, 
                 'RandomForestClassifier' : RandomForestClassifier, 
                 'LogisticRegression' : LogisticRegression, 
                 'KNeighborsClassifier' : KNeighborsClassifier, 
                 'DecisionTreeClassifier' : DecisionTreeClassifier,
                 'GaussianNB' : GaussianNB
                 }
    
def best_estimator(model_name):
    
    if model_name != 'NN': _check_name(model_name)
    
    _, model_path = best_results()
    model, config = get_1estimator(model_path[model_name], model_name)
    
    return model, config, model_path[model_name]

def get_1estimator(file_path, model_name = 'NN'):
    
    if model_name == 'NN':        
        return  _neural_1estimator(file_path)
    
    else:
        _check_name(model_name)
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
    
        row = next(r  for _, r  in df.iterrows() 
                   if model_name in r['estimator'])
    
        txt = row['estimator'].split('(')[1].split(')')[0].replace(' ', '')
        txt = txt.replace('\n', '')
        txt = txt.replace("'", '')
    
        hyper_params = {}
    
        for hyper_value in txt.split(','):
            x =  hyper_value.split('=')
            try: hyper_params[x[0]] = _parser(x[1])
            except: pass
         
        _update_estimator_params(df.columns, hyper_params, row)

        return estimator_dic[model_name](**hyper_params), hyper_params

def _neural_1estimator(file_path):
    
    df = pd.read_csv(file_path)
    hyper_params = {}
    _update_estimator_params(df.columns, hyper_params, df.iloc[0])

    early_stop = EarlyStopping(monitor='loss', patience= 3, min_delta=0.001)
    
    return KerasClassifier(build_nn, epochs = 1000, 
                           callbacks = [early_stop],
                           **hyper_params), hyper_params

def _check_name(model_name):
    if model_name not in estimator_dic:
        raise Exception("Model '{}' {}. Choose one of the following: NN, {}."
                        .format(model_name, 
                                "not available in the dictionary",
                                ', '.join([key for key in estimator_dic])))
    
    
def _update_estimator_params(columns, dictionary, row):
    
    for col in [x for x in columns if 'estimator__' in x]:
        
        param_name = col.split('__')[1]		
        
        if row[col] != '': 
            try: dictionary[param_name] = _parser(row[col])
            except: pass
            
    return None
            
def _parser(value):    
    
    try: 
        x = float(value)
        y = int(x)
        return y if np.abs(x - y) < 1e-6 else x
    except:
        if value == 'True':
            return True
        elif value == 'False':
            return False
        elif value == 'deprecated':
            raise Exception("Deprecated parameter or value!")
        elif value == 'None':
            return None
        else:
            return value