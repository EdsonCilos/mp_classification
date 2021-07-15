# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:03:12 2021

@author: scien
"""
#Standard packages 
import os
import numpy as np
import pandas as pd

#Graphics packages
from matplotlib import pyplot as plt

#Project packages
from utils import load_encoder
from baseline import als
import final_classifier
import config

blue = (1/255,209/255,209/255)
purple = (90/255, 53/255, 182/255)

"""
Beta version!
This module must be uncoupled from baseline correction!
"""

def test(top_models = 3):
    
    horizontal = [int(x.split('.')[0]) for x in df.columns.values]
    
    y = np.round(100*model.predict_proba(df), 2)
    #pred = encoder.inverse_transform(np.argmax(y , axis = 1))
    idx = y.argsort()[:,::-1][:,:top_models]
    
    label_path = os.path.join(config._get_path('graphics'), 'labeling')
    
    min_array = []
    mean_array = []
    max_array = []
    
    for cl in range(len(encoder.classes_)):
        min_array.append(np.min(all_y[cl]))
        mean_array.append(np.mean(all_y[cl]))
        max_array.append(np.max(all_y[cl]))
        
    
    if not os.path.exists(label_path):
        os.makedirs(label_path)
        
    #https://github.com/matplotlib/matplotlib/issues/8519#issuecomment-608434198        
    plt.ioff()
    
    for i in range(idx.shape[0]):
        
        sample = df.iloc[i]
        k = 1
        plt.figure(figsize=(top_models*5, 7))
            
        for j in idx[i]:
            
             m = encoder.inverse_transform([j])[0]
                         
             plt.subplot(1, top_models, k)
        
             plt.title("{}, Probability: {}%".format(m, y[i][j]))
             plt.xlabel("Wavelenght (1/cm)")
             
             plt.plot(horizontal, sample.values, '-', 
                      color= purple, label= "Sample")
                          
             plt.plot(horizontal, mean_array[j], '-', color= blue, label= m)
             plt.fill_between(horizontal, min_array[j], max_array[j], 
                              alpha= 0.25,color=blue) 
             plt.legend(loc="best")
        
             k += 1
        
        sample = None             
            
        plt.savefig(os.path.join(label_path, 'prediction_{}.png'.format(i)),
                    dpi = 300, 
                    bbox_inches = "tight")
        plt.close()

    return None



def random_analysis():
    sample = df.sample(n=1)
    y = model.predict_proba(sample)
    idx = (-y).argsort()[:3][0][:3]
    
    for i in idx:
        compare(sample, i, np.round(y[0][i]*100, 2))

def plot_all_classes():
    
    data, encoder = load_data()
          
    for mp_class in range(14):
        plot_class(mp_class)


def compare(sample, y, score):
    m = encoder.inverse_transform([y])[0]
    
    plt.figure(figsize=(7, 7))    
    plt.title("{}, Probability: {}%".format(m, score))
    plt.xlabel("Wavelenght (cm-1)")
    #plt.ylim(bottom=0, top = 0.6)
    
    
    horizontal = [int(x.split('.')[0]) for x in sample.columns.values]

    plt.plot(horizontal, sample.values[0], '-', color= purple, label= "Sample")
    
    data_y = baseline_sample(data, y)
    
    min_array = np.min(data_y)
    mean_array = np.mean(data_y)
    max_array = np.max(data_y)
    
    plt.plot(horizontal, mean_array, '-', color= blue, label= m)
    plt.fill_between(horizontal, min_array, max_array, alpha= 0.25, color=blue)

    plt.legend(loc="best")

    plt.show()

    return None

def plot_class(mp_class):
    
    plt.figure(figsize=(7, 7))
    
    plt.title(encoder.inverse_transform([mp_class])[0])
    plt.xlabel("Wavelenght (cm-1)")
    #plt.ylim(bottom=0, top = 0.6)
    
    new_data = baseline_sample(data, mp_class)
       
    min_array = np.min(new_data)
    mean_array = np.mean(new_data)
    max_array = np.max(new_data)
    
    
    horizontal = [int(x.split('.')[0]) for x in mean_array.index.values]

    # Plot learning curve
    plt.fill_between(horizontal, min_array, max_array, alpha= 0.25, color= blue)

    plt.plot(horizontal, mean_array, '-', color= blue, label="Signal")

    plt.legend(loc="best")

    plt.show()

    return None


def load_data_unlabeled():
    return  pd.read_csv(os.path.join('data', 'baseline_unlabeled_data.csv'))


def load_data():
    
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
    X_test = pd.read_csv(os.path.join('data', 'X_test.csv')) 
    y_test = pd.read_csv(os.path.join('data', 'y_test.csv')).values.ravel()
    
    data = pd.concat([X_train, X_test])
    data['label'] = pd.Series(np.concatenate((y_train, y_test)), 
                             index= data.index)
    
    return data, load_encoder()

def baseline_sample(mp_class, data):
    
    new_data = data[data['label'] == mp_class].copy()
    new_data.drop(['label'], axis=1, inplace=True)
    new_data.reset_index(drop = True, inplace = True)
    
    for idx, row in new_data.iterrows():
         new_data.iloc[idx, :] = row - als(row)
         
    return new_data
    


def plot_sample(sample):
    horizontal = [int(x.split('.')[0]) for x in sample.columns.values]
    values = sample.values[0]
    plt.plot(horizontal, values, 'r')
    plt.plot(horizontal, values - als(values), 'b')
    plt.show()

def run():
    
    df = load_data_unlabeled()
    data, encoder = load_data()
    model = final_classifier.load_model()
    
    print("Generating baseline for each class...")
    all_y = [baseline_sample(y, data) 
             for y in range(len(encoder.classes_))]
    print("Baseline complete...")
    
    return df, data, encoder, model, all_y

df, data, encoder, model, all_y = run()