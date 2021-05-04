#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:51:22 2020

@author: Edson Cilos
"""
import pickle
import pandas as pd
import numpy as np
import os
from os import path
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from timeit import default_timer as timer

#load project modules
from param_grid import build_nn
from grid_search import append_time
from utils import build_row

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Beta version!

def mccv_path():
    return os.path.join('results', 'mccv')

def results_total(X, name, sufix, temp=True): #Arrumar!
 
   posfix = '_temp' if temp else '' 
    
   filepath = os.path.join(mccv_path(), name, sufix +  posfix + '.csv')
   
   pd.DataFrame(data = X, columns= ['Cross_Entropy_train', 
                                  'Cross_Entropy_val',
                                  'Accuracy_train',
                                  'Accuracy_val']).to_csv(filepath,index=False)

def results(X, name, sufix, temp=True):
    3
    posfix = '_temp' if temp else '' 
    
    filepath = os.path.join(mccv_path(), name, sufix + posfix + '.csv')
   
    pd.DataFrame(data = X).to_csv(filepath, header=False, index=False)



do = True

while(do):
    
    simulate = False
    
    print("Choose an option: \n \n 0 - Leave \n 1- SVC \n 2 - LR \n 3 - NN \n")
    n= int(input("input: "))
    
    base_model = None
    
    name = ''
    
    if(n == 0):
        input("Press any key to finish...")
        break
    
    elif(n == 1):
        
        print("Support Vector Machine selected")
        kernel = input("Type the kernel name: ")
        c = float(input("Type the regularization parameter C: "))
        base_model = SVC(kernel = kernel, C = c, probability = True)
        name =  "_".join(["SVC", kernel, str(c)])
        simulate = True
        
    elif(n == 2):
        
         print("Logistic Regression selected \n")
         c = float(input("Type the regularization parameter C: "))
         
         base_model = LogisticRegression(C=c, solver='newton-cg')

         name =  "_".join(["LR", str(c)])
         simulate = True
         
    elif(n == 3):
        
         print("Neural Network Selected selected \n")
         layers = int(input("Type the number of hidden layers: \n"))
         neurons = int(input("Type the number of neurons: \n"))
         momentum = float(input("Momentum value: \n"))
         lr = float(input("Learning rate: \n"))
         m = int(input("Select the activation function: \n 1 - Sigmoid \n \n 2 - Tanh \n"))
         
         activation = ''
         go = False
         
         if(m == 1):
             activation = 'sigmoid'
             go = True
             
         elif(m == 2):
             activation = 'tanh'
             go = True
         
         if(go):
            
            params = {'n_hidden' : layers,
                      'n_neurons' : neurons, 
                      'momentum' : momentum, 
                      'learning_rate' : lr, 
                      'act' : activation
                      }

            
            base_model = KerasClassifier(build_nn, 
                                    epochs = 1000,
                                    callbacks = [EarlyStopping(monitor='loss', 
                                                            patience= 3,
                                                            min_delta=0.001
                                                            )],
                                    **params
                                 )
            
            
            name_list = [activation] + [str(x) for x in 
                                              [layers, neurons, momentum, lr]]
            
            name =  "_".join(name_list)
            
            simulate = True         
        
    else:
        print("Option not available")
        
    if(simulate):
        
        sv_filter = bool(int(input(
            "Use Savitzky-Golay filter? (0 - no, 1 - yes) \n")))
        pca = bool(int(input("Use pca?  (0 - no, 1 - yes) \n")))
        over = bool(int(input("Use Oversample?  (0 - no, 1 - yes) \n")))
        std = bool(int(input("Use Standard Scaler?  (0 - no, 1 - yes) \n")))
        
        prefix = 'nn_' if n == 3 else ''
        sv = 'svfilter_' if sv_filter else '' 
        sc = 'std_' if std else ''
        pc = 'pca_' if pca else ''
        ov = 'over_' if over else ''
        
        name = prefix + sv + sc +  pc + ov + name
        
        print('Simulation {}'.format(name))
        
        
        encoder_path = os.path.join('data', 'enconder.sav')
        loaded_model = pickle.load(open(encoder_path, 'rb'))
        
        total_score = []
        probability = [] #flattened
        cross_matrix = [] #flattened
        detailed_score = [] #flattened
        
        
        #data has header!
        train_path = os.path.join('data', 'X_train.csv')
        X_train_temp= pd.read_csv(train_path) 
        
        train_path = os.path.join('data', 'y_train.csv')
        y_train_temp= pd.read_csv(train_path).values.ravel()
        
        b = False
        
        folder = os.path.join(mccv_path(), name)
        ts = os.path.join(folder, "total_score_temp.csv")
        prob = os.path.join(folder, "probability_temp.csv")
        cm = os.path.join(folder, "cross_matrix_temp.csv")
        ds = os.path.join(folder, "detailed_score_temp.csv")
        
        m = 0
        
        if not os.path.exists(folder):
            b = True
            os.makedirs(folder)
    
        else: 
            if(path.exists(ts) and path.exists(prob) and path.exists(ds)
            and path.exists(cm)): 
                
                total_score = pd.read_csv(ts).values.tolist()
                probability = pd.read_csv(prob, header=None).values.tolist()
                cross_matrix = pd.read_csv(cm, header=None).values.tolist()
                detailed_score = pd.read_csv(ds, header=None).values.tolist()
                
                
                m = min([len(total_score), 
                         len(probability), 
                         len(cross_matrix),
                         len(detailed_score)])
    
                if (m > 0):
                    total_score = total_score[:m]
                    probability = probability[:m]
                    cross_matrix = cross_matrix[:m] 
                    detailed_score = detailed_score[:m]
                    print("Backup Loaded! \n")
                    print( str(m) + " iterarions restored")
                else:
                    total_score = []
                    probability = []
                    cross_matrix = []
                    detailed_score = []
            else:
                
                b = True
                
        if(b): 
            results_total(total_score, name, 'total_score')
            results(cross_matrix, name, 'cross_matrix')
            results(probability, name, 'probability')
            results(detailed_score, name, 'detailed_score')
            
        start = timer()
                
        for i in range(m, 1700): 
    
            print("Iteration number: " + str(i) + "---------" + name)
    
            if(i%100 == 0): #Backup to avoid loose all the job!
    
            	try:
                    
                    results_total(total_score, name, 'total_score')
                    results(cross_matrix, name, 'cross_matrix')
                    results(probability, name, 'probability')
                    results(detailed_score, name, 'detailed_score')
                    
                    print("New backup version saved")
                    
            	except: 
                      print("Backup failed")
                      
            
            X_train, X_val, y_train, y_val = train_test_split(X_train_temp, 
                                                      y_train_temp,
                                                      stratify=y_train_temp, 
                                                      test_size= 1/3)
    
            if(sv_filter):
                X_train = savgol_filter(X_train, 11, 3)
                X_val = savgol_filter(X_val, 11, 3)
                
            if(std):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                
            if(pca):
                pca = PCA(n_components = 0.99)
                X_train = pca.fit_transform(X_train)
                X_val = pca.transform(X_val)
                
            if(over):
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_resample(X_train, y_train)
                
            
            model = clone(base_model)
            model.fit(X_train, y_train)
            
            predict_array = model.predict(X_val)
            predicted_prob = model.predict_proba(X_val)
                
            total_score.append(
                    [log_loss(y_train, model.predict_proba(X_train)),
                     log_loss(y_val, predicted_prob),
                     accuracy_score(np.array(y_train), model.predict(X_train)),
                     accuracy_score(np.array(y_val), predict_array)
                     ])
            
            flatten_probabilities = []
            
            flatten_cross = []
            
            for j in range(14):
                
                idxs = np.where(y_val == j)[0] #idxs of true j-label
                
                flatten_probabilities.extend(
                        np.mean(predicted_prob[idxs], axis = 0)  #class probs
                        )
                
                flatten_cross_by_class = [
                        len(np.where(predict_array[idxs] == k)[0]) 
                        for k in range(14) 
                        ]
            
                    
                flatten_cross.extend(flatten_cross_by_class)
                
            
            cross_matrix.append(flatten_cross)
            
            probability.append(flatten_probabilities)
            
            detailed_score.append(build_row(X_val, y_val, predict_array))
            
            #Finish loop
        
        end = timer()
        append_time("MCCV/" + name, str(end - start))
        print("Finishing job....")
        print("Saving....")
        results_total(total_score, name, 'total_score', temp=False)
        results(cross_matrix, name, 'cross_matrix', temp=False)
        results(probability, name, 'probability', temp=False)
        results(detailed_score, name, 'detailed_score', temp=False)
        print("Deleting temporary files....")
        os.remove(ts)
        os.remove(prob)
        os.remove(cm)
        os.remove(ds)
        print("job " + name + " done with success!")
        