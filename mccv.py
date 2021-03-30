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
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from utils import build_row

#load project modules
from neural_search import build_model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def results_total(X, name, sufix, temp=True): #Arrumar!
 
   posfix = '_temp' if temp else '' 
    
   filepath = os.path.join(os.getcwd(),  
                           'mccv_data',
                           name,
                           sufix +  posfix + '.csv')
   
   pd.DataFrame(data = X, columns= ['Cross_Entropy_train', 
                                  'Cross_Entropy_val',
                                  'Accuracy_train',
                                  'Accuracy_val']).to_csv(filepath,index=False)

def results(X, name, sufix, temp=True):
    
    posfix = '_temp' if temp else '' 
    
    filepath = os.path.join(os.getcwd(),  
                           'mccv_data',
                           name,
                           sufix + posfix + '.csv')
   
    pd.DataFrame(data = X).to_csv(filepath, header=False, index=False)



do = True

while(do):
    
    simulate = False
    
    print("Choose an option: \n \n 0 - Leave \n 1- SVC \n 2 - RF \n 3 - NN \n")
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
        
         print("Random Forest selected \n")
         estimators = int(input("Type the number of estimators: \n"))
         base_model = RandomForestClassifier(n_estimators=estimators, 
                                        criterion = 'entropy')

         name =  "_".join(["RF", str(estimators)])
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

            
            base_model = KerasClassifier(build_model, 
                                    epochs = 1000,
                                    callbacks = [EarlyStopping(monitor='loss', 
                                                            patience= 3,
                                                            min_delta=0.001
                                                            )],
                                    **params
                                 )
            
            
            name_list = ['NN', activation] + [str(x) for x in 
                                              [layers, neurons, momentum, lr]]
            
            name =  "_".join(name_list)
            
            simulate = True         
        
    else:
        print("Option not available")
        
    if(simulate):
        
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
        
        folder = os.path.join('mccv_data', name)
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
                
        for i in range(m, 17000): 
    
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
    
            scaler = StandardScaler()
            X_train  = scaler.fit_transform(savgol_filter(X_train, 11, 10))
            
            X_val = scaler.transform(savgol_filter(X_val, 11, 10))
    
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
        
        
        
                
            #Depois usar na documentação (via doc-string)
            #salvar resultados da probabilidade!
    

            #File 1: log_loss_train, log_loss_val, accuracy_train, accuracy_val
            #Cada linha corresponde a um split
            
            
            #File 2: Proability multy-matrix
            #Colunas (196 = 14*14 ao todo)
            #p_(0,0), p_(0,1), ..., p_(0,13), p_(1,0), ....., p(13,13)
            
            #p_(i,j) =
            #probabilidade média atribuiada pelo algoritmo class
            #de amostras da classe i como sendo da classe j
            #Idealmente a matriz 14x14 deveria ser aproximadamente a identidade
        
            
            #linhas: resultado de cada split
            
            #File 3: (sensitivity, specificity, precision, f1_score) 
            #  Arquivo csv com 4*14 colunas
            # Cada linha corresponde a um split
            