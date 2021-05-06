# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 23:57:33 2021

@author: Edson Cilos
"""

#Standard Modules
import os
import glob
import numpy as np
import pandas as pd

#sklearn models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#To save models
import pickle
#Config module
import config


seed = config._seed()

def check_in(values_list, name):
    for value in values_list:
        if value in name: 
            return False
    return True

def build_data():
    
    i=0
    d4_rebuild= pd.read_csv(os.path.join('data', 'd4_rebuild.csv'))
    exclude = [get_path(row[1]['Nom ']) for row in d4_rebuild.iterrows()]

    #Just to get the columns (supposing that all data in the same format)
    especific = os.path.join('data', 'IR_Spectra', '300914', 'M204', '500', 
                             'TM0033D1.txt')

    columns = np.round(pd.read_table(especific, header=None, sep=r'\s+')[0],5)
    
    instances = []


    for path in glob.iglob(os.path.join('data', 'IR_Spectra') + '**/**', 
                           recursive=True):
    
        if not os.path.isdir(path) and check_in(exclude, path):
            try:
                data = pd.read_table(path, header = None, sep=r'\s+')
                instances.append(data[1].values)
            except:
                print("Not possible to read file: "  + path)
                
    print(i)

    df = pd.DataFrame(data = instances, columns = columns.values)
    df.to_csv(os.path.join('data', 'unlabeled_data.csv'), index = False)
    
    return df
    
def restruct_d4():
    
    df = pd.read_csv(os.path.join('data', 'D4_4_publication.csv'))
    columns = df.columns
    instances = []
    
    for row in df.iterrows():
        
        file_path = get_path(row[1]['Nom '])
    
        try:
            data = pd.read_table(file_path, header = None, sep=r'\s+')
            instances.append([row[1]['Nom '], row[1]['Interpretation ']] 
                         + list(data[1].values[1:]))
        except:
            print("Problem in the sample: {} - {}"
                  .format(row[1]['Nom '], row[1]['Interpretation ']))
            
    
            
    df = pd.DataFrame(data = instances, columns = columns.values)
    df.to_csv(os.path.join('data', 'd4_rebuild.csv'), index = False)
    
    return df

def get_path(sample_code):
    
     name = sample_code.split('_')
     
     prefix = name[0][:-1] if name[0][-1] in ['A', 'B'] else name[0]
     
     if prefix[0] != 'M':
         prefix = 'SA' if prefix[0] == 'S' else 'M' + prefix
            
     return os.path.join('data', 'IR_Spectra', name[1], prefix, name[2], 
                         name[3] + ".txt")
     
            
def pre_process(threshold = 5, override = False):
      
  dataset = pd.read_csv(os.path.join(os.getcwd(), "data", "d4_rebuild.csv"))
  dataset.drop(['Nom '], axis=1, inplace=True)
  dataset.rename(columns={'Interpretation ': "label"}, inplace=True)
  
  freq = dataset["label"].value_counts()
  
  print('Previous class frequency:') 
  print(freq)
  
  less_rep = [idx for idx, value in freq.items()  if value < threshold]
  
  for i, row in dataset.iterrows():
      if row["label"] in less_rep:
          dataset.at[i, 'label'] = 'Unknown'

  print('\n \n \n New class frequency (after removing Unkown):') 
  print(dataset["label"].value_counts())
  
  encoder = LabelEncoder()
  dataset["label"] = encoder.fit_transform(dataset["label"])
  
  
  
  X = dataset.drop(["label"], axis = 1)
  y = dataset["label"].copy()
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                      shuffle=True,
                                                      test_size=1/4,
                                                      random_state = seed)
  
  if(override):
      print('\n \n \n Be careful when overriding files!')
      #Savedatasets to allow reproducibility
      X_train.to_csv(os.path.join("data", "X_train.csv"), index=False)
      y_train.to_csv(os.path.join("data", "y_train.csv"), index=False)
      X_test.to_csv(os.path.join("data", "X_test.csv"), index=False)
      y_test.to_csv(os.path.join("data", "y_test.csv"), index=False)
      
      #save encoder
      pickle.dump(encoder, open(os.path.join("data", 'enconder.sav'), 'wb'))

  return dataset, X_train, X_test, y_train, y_test, encoder
    
