# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:09:51 2021

@author: Edson Cilos
"""
#Sklearn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier

#Imblearn modules
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline 

#Config file
import config

#Fix seed to reproducibility
seed = config._seed() 
        
    
def build_pipe(scaler = '', 
               baseline = True,
               pca = True, 
               over_sample = True):
    
    prefix = 'baseline_' if baseline else ''
    
    pre_pipe =[('estimator', DummyClassifier())]
       
    
    scaler_dictionary = {
        'std' : StandardScaler(), 
        'minmax' : MinMaxScaler()
        }
    
    if(scaler in scaler_dictionary):
        
        prefix += scaler + '_'    
        pre_pipe.insert(-1, ('scaler', scaler_dictionary[scaler]))         
                
    if(pca):        
        prefix += 'pca_'        
        pre_pipe.insert(-1,
                        ('dim_red', 
                         PCA(n_components = 0.99, random_state = seed)
                         ))
    if(over_sample):
        
       prefix += 'over_'
        
       pre_pipe.insert(-1, ('over_sample',
                            RandomOverSampler(random_state = seed)) 
                             )

    return  Pipeline(pre_pipe), prefix

def pipe_config(file_name):
    
    values = file_name.split('_')
    
    scaler_list = [x for x in config._scaler_list() if x in values]
    
    try:
        scaler = scaler_list[0]
    except:
        scaler = ''
    
    baseline = True if 'baseline' in values else False
    pca = True if 'pca' in values else False
    over_sample = True if 'over' in values else False
    
    return baseline, scaler, pca, over_sample
  
    