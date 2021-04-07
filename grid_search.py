#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:52:03 2020

@author: Edson Cilos
"""
#Standard modules
import os
import pandas as pd
import numpy as np
from itertools import product

#Filter preprocessing
from scipy.signal import savgol_filter

#Sklearn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

#Imblearn modules
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline 

#Project modules
from neural_search import neural_grid

#Fix seed to reproducibility
seed = 0

class Savgol_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, window = 11, degree = 10):
        assert window > degree, "window must be less than poly. degree"
        self.window = window
        self.degree = degree
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None ):
        return savgol_filter(X, self.window, self.degree)
    
def build_pipe(scaler = '', 
               sav_filter = True,
               pca = True, 
               over_sample = True):
    
    prefix = ''
    
    pre_pipe =[('estimator', DummyClassifier())]
    
    if sav_filter:
        prefix += 'svfilter_'
        pre_pipe.insert(-1, ('filter', 
                             Savgol_transformer(window = 11, degree=10)))        
    
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

def base_dic(estimator):
    return {'estimator': [estimator]} 

#Basic grid structure for classical algorithms, expecpt neural network
def classical_grid():
    
     #DecisionTree
    decision_tree = base_dic(DecisionTreeClassifier(random_state = seed))
    decision_tree['estimator__criterion'] = ['gini', 'entropy']
    decision_tree['estimator__splitter'] = ['best', 'random']
    
    #NaiveBayes
    gaussian_nb = base_dic(GaussianNB())

    #Knn
    knn = base_dic(KNeighborsClassifier())
    knn['estimator__n_neighbors'] = [1, 3, 5, 7, 9, 10, 11, 13]
    knn['estimator__weights'] = ['uniform', 'distance']
    knn['estimator__p'] = [1, 2, 3, 4]

    #Random_forest
    random_forest = base_dic(RandomForestClassifier(random_state = seed))
    random_forest['estimator__n_estimators'] = [10, 100, 1000]
    random_forest['estimator__criterion'] = ['gini', 'entropy']
    
    #Parameter C, used in several models    
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    #Losgitic Regression with l2 penalty
    logistic_regression_1 = base_dic(LogisticRegression(
        solver='newton-cg',
        penalty='l2',
        multi_class = 'auto',
        random_state = seed))
    logistic_regression_1["estimator__C"] = c_list.copy()

    #Logistic Regression with l1 penalty
    logistic_regression_2 = base_dic(LogisticRegression(
        solver = 'liblinear',
        penalty = 'l1',
        multi_class='auto',
        random_state = seed))
    logistic_regression_2["estimator__C"] = c_list.copy()

    #Losgitic Regression with elasticnet penalty   
    logistic_regression_3 = base_dic(LogisticRegression(
        solver = 'saga',
        penalty = 'elasticnet',
        multi_class='auto',
        random_state = seed))
    logistic_regression_3['estimator__l1_ratio'] = np.arange(0.1, 0.9, 0.1)
    logistic_regression_3["estimator__C"] = c_list.copy()
    
    svc_1 = base_dic(SVC(kernel = 'linear', 
                         probability=True, 
                         random_state = seed))
    
    svc_1["estimator__C"] = c_list.copy()

    svc_2 = base_dic(SVC(probability=True, random_state = seed))
    svc_2["estimator__C"] = c_list.copy()
    svc_2["estimator__kernel"] = ['rbf', 'poly', 'sigmoid']
    svc_2["estimator__gamma"] = ['scale', 'auto']

    return [decision_tree, gaussian_nb, knn, random_forest,
            logistic_regression_1, logistic_regression_2, 
            logistic_regression_3, svc_1, svc_2]

    
    
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
        sc = 'std' if scaler else ''
        grid = neural_grid() if nn else classical_grid()
        prefix = 'nn_' if nn else ''
        
        file_name = prefix + 'svfilter_' if sv_filter else '' + sc +  \
        'pca_' if pca else '' + 'over_' if over else '' + 'gs.csv'
        
        file_path = os.path.join(os.getcwd(), 'results', file_name)
        
        if os.path.isfile(file_path):
            print(file_name + " already exists, iteration was skipped ...")
            
        else:
            print("{0} iteration ({1}/32)...".format(file_name, str(i)))
            search(scaler = sc, 
                   sav_filter = sv_filter,
                   pca = pca, 
                   over_sample = over, 
                   param_grid = grid, 
                   prefix = prefix)
            