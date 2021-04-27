# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:50:53 2021

@author: Edson cilos
"""

#Standard Packages 
import numpy as np

#Sklearn API
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Tensorflow API
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping


#Fix seed to reproducibility
seed = 0

#Be aware about Keras's issue https://github.com/keras-team/keras/issues/13586
#Solution here https://stackoverflow.com/questions/62801440/kerasregressor-cannot-clone-object-no-idea-why-this-error-is-being-thrown/66771774#66771774




#Basic setup to build neural network
def build_nn(n_hidden = 1, 
             n_neurons = 50, 
             momentum = 0.9,
             learning_rate = 0.001, 
             act = "sigmoid"):
    
    model = keras.models.Sequential()
    
    for layer in range(int(n_hidden)):
        model.add(keras.layers.Dense(n_neurons, activation= act))
        
    model.add(keras.layers.Dense(14, activation="softmax"))
    
    optimizer = keras.optimizers.SGD(momentum = momentum,
                                     nesterov = True, 
                                     lr=learning_rate)
    
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    return model


def neural_grid(epochs = 1000, patience = 3):

    #Create dictionary with base model, including parameter grid
    neural_network = {'estimator': [KerasClassifier(build_nn, 
                                 epochs = epochs,
                                 callbacks = [EarlyStopping(monitor='loss', 
                                                            patience= patience,
                                                            min_delta=0.001
                                                            )]
                                 )],
        "estimator__n_hidden": [1, 2, 3, 4, 5],
        "estimator__n_neurons": [10, 50, 100, 150, 200],
        "estimator__momentum" : np.arange(0.1, 0.9, 0.3),
        "estimator__learning_rate": [1e-3, 1e-2, 1e-1],
        "estimator__act": ["relu", "sigmoid", "tanh"],
        }
    
    return [neural_network]

def filter_grid(estimator, max_window = 11):
    
    return [{'estimator': [estimator],
             'filter__window': [window],
             'filter__degree': np.arange(2, window)
             }  for window in range(3, max_window + 1, 2) ]

#Basic grid structure for classical algorithms, expecpt neural network
def classical_grid():
    
    
    decision_tree = {'estimator':[DecisionTreeClassifier(random_state = seed)],
                     'estimator__criterion':  ['gini', 'entropy'],
                     'estimator__splitter': ['best', 'random']
                     }
    
    #NaiveBayes
    gaussian_nb = { 'estimator': [GaussianNB()] }

    #Knn
    knn = {'estimator': [KNeighborsClassifier()],
           'estimator__n_neighbors' : [1, 3, 5, 7, 9, 10, 11, 13],
           'estimator__weights' : ['uniform', 'distance'],
           'estimator__p' : [1, 2, 3, 4]
           }
    
    #Random_forest
    random_forest ={'estimator': [RandomForestClassifier(random_state = seed)],
                    'estimator__n_estimators' : [10, 50, 100, 500],
                    'estimator__criterion' : ['gini', 'entropy']
    }
    
    #Parameter C, used in several models    
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    #Losgitic Regression with l2 penalty
    logistic_regression_1 = {'estimator': [LogisticRegression(
                                            solver='newton-cg',
                                            penalty='l2',
                                            multi_class = 'auto',
                                            random_state = seed)],
                              'estimator__C' : c_list.copy()
                              }
    

    #Logistic Regression with l1 penalty
    logistic_regression_2 = {'estimator': [LogisticRegression(
                                            solver = 'liblinear',
                                            penalty = 'l1',
                                            multi_class='auto',
                                            random_state = seed)],
                            'estimator__C' : c_list.copy()
                            }

    #Losgitic Regression with elasticnet penalty
    logistic_regression_3 = {'estimator': [LogisticRegression(
                                            solver = 'saga',
                                            penalty = 'elasticnet',
                                            multi_class='auto',
                                            random_state = seed)],
                            'estimator__l1_ratio': np.arange(0.1, 1, 0.1),
                            'estimator__C' : c_list.copy()
                            }

    #Suppport vector machine with linear kernel    
    svc_1 = {'estimator': [SVC(kernel = 'linear', probability=True, 
                               random_state = seed)],
             'estimator__C' : c_list.copy() 
             }
    
    #Suppport vector machine with non-linear kernel
    svc_2 = {'estimator': [SVC(probability=True, random_state = seed)],
             'estimator__C' : c_list.copy(),
             'estimator__kernel' : ['rbf', 'poly', 'sigmoid'],
             'estimator__gamma' : ['scale', 'auto']
        }

    return [decision_tree, gaussian_nb, knn, random_forest,
            logistic_regression_1, logistic_regression_2, 
            logistic_regression_3, svc_1, svc_2]
