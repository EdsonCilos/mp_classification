#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:52:03 2020

@author: edson
"""
import os
import pandas as pd
from scipy.signal import savgol_filter
#Sklearn modules
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

#Fix seed to reproducibility
seed = 0

class Savgol_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, window, degree):
        assert window > degree, "window must be less than poly. degree"
        self.window = window
        self.degree = degree
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None ):
        return savgol_filter(X, self.window, self.degree)
    
def grid_search(pca = True, over_sample = True):
    
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
        
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
    file_name = ''
    
    pre_pipe =[('filter', Savgol_transformer(11, 10)),
             ('scaler', StandardScaler()),
             ('estimator', DummyClassifier())
             ]
    
    if(pca):
        
        file_name += 'pca_'
        
        pre_pipe.insert(2, ('dim_red', 
                            PCA(n_components = 0.99, random_state = seed)
                            ))
        
    if(over_sample):
        
       file_name += 'over_'
        
       pre_pipe.insert(-1, ('over_sample',
                            RandomOverSampler(random_state = seed)) 
                             )
        
    print(pre_pipe)
    
    pipe = Pipeline(pre_pipe)
    

        
        
    def base_dic(estimator):
        return {'estimator': [estimator]}


    decision_tree = base_dic(DecisionTreeClassifier(random_state = seed))
    decision_tree['estimator__criterion'] = ['gini', 'entropy']
    decision_tree['estimator__splitter'] = ['best', 'random']

    gaussian_nb = base_dic(GaussianNB())


    knn = base_dic(KNeighborsClassifier())
    knn['estimator__n_neighbors'] = [1, 3, 5, 7, 9, 10, 11, 13]
    knn['estimator__weights'] = ['uniform', 'distance']
    knn['estimator__p'] = [1, 2, 3, 4]


    random_forest = base_dic(RandomForestClassifier(random_state = seed))
    random_forest['estimator__n_estimators'] = [10, 100, 1000]
    random_forest['estimator__criterion'] = ['gini', 'entropy']
    
    logistic_regression_1 = base_dic(LogisticRegression(random_state = seed))
    logistic_regression_1['estimator__solver'] = ['newton-cg']
    logistic_regression_1['estimator__penalty'] = ['l2']
    logistic_regression_1['estimator__multi_class'] = ['auto']
    logistic_regression_1["estimator__C"] = [0.001, 0.01, 0.1,  \
                         1, 10, 100, 1000]


    logistic_regression_2 = base_dic(LogisticRegression(random_state = seed))
    logistic_regression_2['estimator__solver'] = ['liblinear']
    logistic_regression_2['estimator__penalty'] = ['l1']
    logistic_regression_2['estimator__multi_class'] = ['auto']
    logistic_regression_2["estimator__C"] = [0.001, 0.01, 0.1, \
                         1, 10, 100, 1000]

    logistic_regression_3 = base_dic(LogisticRegression(random_state = seed))
    logistic_regression_3['estimator__solver'] = ['saga']
    logistic_regression_3['estimator__penalty'] = ['elasticnet']
    logistic_regression_3['estimator__multi_class'] = ['auto']
    logistic_regression_3['estimator__l1_ratio'] = [0.1, 
                                                0.2, 
                                                0.3, 
                                                0.4, 
                                                0.5, 
                                                0.6, 
                                                0.7, 
                                                0.8, 
                                                0.9]
    
    logistic_regression_3["estimator__C"] = [0.001, 0.01, 0.1,  1,\
                         10, 100, 1000]
    


    svc_1 = base_dic(SVC(probability=True, random_state = seed))
    svc_1["estimator__C"] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    svc_1["estimator__kernel"] = ['linear']

    svc_2 = base_dic(SVC(probability=True, random_state = seed))
    svc_2["estimator__C"] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    svc_2["estimator__kernel"] = ['rbf', 'poly', 'sigmoid']
    svc_2["estimator__gamma"] = ['scale', 'auto']

    param_grid = [decision_tree, 
                  gaussian_nb,
                  knn, 
                  random_forest,
                  logistic_regression_1, 
                  logistic_regression_2, 
                  logistic_regression_3,
                  svc_1,
                  svc_2
                  ]



    cv_fixed_seed = StratifiedKFold(n_splits=4, random_state = seed)

    grid_search = GridSearchCV(pipe,
                               param_grid,
                               scoring = 'neg_log_loss',
                               cv = cv_fixed_seed,
                               n_jobs = -2,
                               verbose = 100)

    grid_search = grid_search.fit(X_train, y_train)


    results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                     pd.DataFrame(grid_search.cv_results_['std_test_score'], 
                                  columns=["std"]),
                     pd.DataFrame(grid_search.cv_results_["mean_test_score"], 
                                  columns=["neg_log_loss"])],axis=1)
                     
    results.sort_values(by=['neg_log_loss'], ascending=False, inplace=True)
        
    file_name += 'gs.csv'
    
    folder = os.path.join(os.getcwd(), 'results')
    
    if not os.path.exists(folder):
            os.makedirs(folder)

    results.to_csv(os.path.join(folder, file_name), index = False)
    
    return results