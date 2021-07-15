# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:55:56 2021

@author: scien
"""
#Standard packages 
import os
import numpy as np
import pandas as pd

#Graphics packages
from matplotlib import pyplot as plt

#ML modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.svm import SVC

#Project packages
from utils import build_row
from baseline import als
import config



mccv_path = config._get_path( 'mccv')
seed = config._seed()


def results_total(X, name, sufix):
 
   filepath = os.path.join(mccv_path + '_newsample', 
                           name, sufix + '.csv')
   
   pd.DataFrame(data = X, columns= ['Cross_Entropy_train', 
                                  'Cross_Entropy_val',
                                  'Accuracy_train',
                                  'Accuracy_val']).to_csv(filepath,index=False)

def results(X, name, sufix):
    filepath = os.path.join(mccv_path + '_newsample', name, 
                            sufix + '.csv')
   
    pd.DataFrame(data = X).to_csv(filepath, header=False, index=False)



path = os.path.join('data','new_sample.csv')
df = pd.read_csv(path,decimal=r',')
df.drop(columns=['Spectrum ID'], inplace=True)
y = df['Polymer'].copy()
X = df.drop(columns=['Polymer'])

freq = df['Polymer'].value_counts()

print('Previous class frequency:') 
print(freq)

threshold = 10

less_rep = [idx for idx, value in freq.items()  if value < threshold]
  
for i, row in df.iterrows():
    if row['Polymer'] in less_rep:
        df.at[i, 'Polymer'] = 'Unknown'

print('\n \n \n New class frequency (after removing Unkown):') 
print(df['Polymer'].value_counts())


encoder = LabelEncoder()
y = encoder.fit_transform(df['Polymer'])


print("Running preprocessing...")
for idx, row in X.iterrows():
    X.iloc[idx, :] = row - als(row)

base_model = SVC(kernel = 'linear', C = 100, 
            probability = True, random_state= seed)

n_class = np.unique(y).shape[0]

total_score = []
probability = [] #flattened
cross_matrix = [] #flattened
detailed_score = [] #flattened

name = 'baseline_SVC_linear_100.0'
        
for i in range(0, 1700): 
    
    X_train, X_val, y_train, y_val = train_test_split(X, 
                                                      y,
                                                      stratify=y, 
                                                      test_size= 1/3)
    
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)
    model = clone(base_model)
    model.fit(X_train, y_train)
    
    predict_array = model.predict(X_val)
    predicted_prob = model.predict_proba(X_val)
    
    total_score.append([log_loss(y_train, model.predict_proba(X_train)),
                     log_loss(y_val, predicted_prob),
                     accuracy_score(np.array(y_train), model.predict(X_train)),
                     accuracy_score(np.array(y_val), predict_array)
                     ])
    
    flatten_probabilities = []
            
    flatten_cross = []
    
    for j in range(n_class):
        
        idxs = np.where(y_val == j)[0] #idxs of true j-label
        
        flatten_probabilities.extend(np.mean(predicted_prob[idxs], axis = 0))
        flatten_cross_by_class = [len(np.where(predict_array[idxs] == k)[0]) 
                        for k in range(n_class) ]
        
        flatten_cross.extend(flatten_cross_by_class)
        
    cross_matrix.append(flatten_cross)
    probability.append(flatten_probabilities)
    detailed_score.append(build_row(X_val, y_val, predict_array))

results_total(total_score, name, 'total_score')
results(cross_matrix, name, 'cross_matrix')
results(probability, name, 'probability')
results(detailed_score, name, 'detailed_score')


names = encoder.inverse_transform([i for i in range(n_class)])

import seaborn as sns

graphics_path = config._get_path('graphics')

def cross_heatmap(df, name):
        
    w = df.mean(axis=0).values.reshape(n_class, n_class)  #ndarray
    
    for i in range(n_class):
        w[i] /= np.sum(w[i])
        
    w = np.around(w, decimals=3)
        
    cross_frame = pd.DataFrame(data = w, columns = names, index = names)	
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(cross_frame, annot=True, linewidths= 1, cmap="YlGnBu", ax = ax)
    ax.set_title('True class v.s. Predicted Class (mean)')
    
    fig.savefig(os.path.join(graphics_path, name + '_cross_prediction.png'),
                dpi = 1200,
                bbox_inches = "tight")
    

def detailed_score_heatmap(df, name):
      
    w = df.mean(axis=0).values.reshape(n_class, 4)
    
    w = np.around(w, decimals=3)
    
    score_frame = pd.DataFrame(data = w,
                      columns=['sensitivity', 'specificity', 
                               'precision', 'f1_score'],
                      index = names)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    #color_map = plt.get_cmap('YlGnBu_r')
    #color_map = ListedColormap(color_map(np.linspace(0.1, 0.6, 256)))

    sns.heatmap(score_frame,
                annot=True, linewidths= 0.05, cmap='YlGnBu', ax = ax)
    
    ax.set_title(name + ' Scores')

    fig.savefig(os.path.join(graphics_path, name + '_detailed_score.png'),
                dpi = 1200,
                bbox_inches = "tight")


"""
def plot_sample(sample):
    horizontal = [int(x.split('.')[0]) for x in sample.columns.values]
    values = sample.values[0]
    plt.xlabel("Wavelenght (1/cm)")
    plt.plot(horizontal, values, color = config._blue, label = 'sample')
    plt.plot(horizontal, values - als(values), 
             color = config._purple, label= 'sample (corrected)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
plot_sample(X.sample(n=1))
"""