#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:02:39 2020

@author: Edson Cilos
"""
#Standard packages 
import os
import numpy as np
import pandas as pd

#Sklearning package
from sklearn.preprocessing import MinMaxScaler

#Graphics packages
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot as qq
from matplotlib.colors import ListedColormap
import seaborn as sns

#Project packages
from utils import classes_names
from table import best_results
from config import _mccv_path


#Still beta, several updates required!

#Best model path: 
best_path = os.path.join('results', 'mccv', 'pca_over_SVC_linear_10.0', 
                    'detailed_score.csv')

mccv_path = _mccv_path()

def qq_plot(data):
    return qq(data, line='s')

def gs_heatmap(output_name = 'gs_table'):
    
    df, _ = best_results()
    
    c_map = plt.get_cmap('YlGnBu')
    c_map = ListedColormap(c_map(np.linspace(0.1, 0.7, 256)))

    fig, ax = plt.subplots(figsize=(12, 7))
    heat = sns.heatmap(df, annot=True, linewidths= 1, 
                       cmap=c_map, ax = ax, fmt='.4f')
    
    #Ad-hoc
    for text in heat.texts:
        txt = text.get_text()
        n = float(txt)
        if(n == 0 or n ==1 ): text.set_text('Yes' if n else 'No')
        

    ax.set_title('Grid Search')
    
    fig.savefig(os.path.join('results', output_name + '.png'),
                dpi = 1200, 
                bbox_inches = "tight")
        
    return df
        

def total_score_plot_all():
    _total_score_plot(mccv_files(), "Best models")
    
    
def _total_score_plot(name_list, main_name):
    
    df_tuples = []

    for name in name_list:
        
        df = pd.read_csv(os.path.join(mccv_path, name,'total_score.csv'))
        
        std = np.std(df[df.columns[1]])
        mean = np.mean(df[df.columns[1]])
        
        #label1 = name
        label1 = name + ' loss: ' + str(round(mean, 5)) + \
            ', std: ' +  str(round(std, 5))
            
        std = np.std(df[df.columns[3]])
        mean = np.mean(df[df.columns[3]])
        
        #label2 = name
        label2 = name + ' score: ' + str(round(mean, 5)) + \
             ', std: ' +  str(round(std, 5))
        
        df_tuples.append((df, label1, label2))
    total_score_plot(df_tuples, main_name)
    
        
def total_score_plot(df_tuples, name):
        
    sns.set_palette(sns.color_palette("hls", len(df_tuples)))
    
    for tup in df_tuples:
        plot = sns.distplot(tup[0]["Cross_Entropy_val"], 
                            axlabel = 'Cross Entropy (validation)',
                            label=tup[1],
                            )

    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.35))

    fig = plot.get_figure()
    
    fig.savefig(os.path.join('results', name + '_cross_entropy.png'), 
                dpi = 1200,
                bbox_inches = "tight")
    
    plt.show()
    plt.close()

    ##The same for accuracy
    sns.set_palette(sns.color_palette("hls", len(df_tuples)))
    
    for tup in df_tuples:
        plot = sns.distplot(tup[0]["Accuracy_val"], 
                            axlabel = 'Accuracy (validation)',
                            label=tup[2])

    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.35))

    fig = plot.get_figure()
    
    fig.savefig(os.path.join('results', name + '_accuracy.png'),
                dpi = 1200,
                bbox_inches = "tight")
    
    plt.show()
    plt.close()
    
def self_heatmap():
    
    df = pd.read_csv(os.path.join('results', 'SelfTraining.csv'), index_col=0)
    df.index.name = None
    df.drop(['base_path'], axis=1, inplace=True)
    
    rename = {'time' : 'Time (s)',
              'amount_labaled' : 'Samples labeled',
              'accuracy' : 'Accuracy',
              'log_loss' : 'Log-los',
              'std_log_loss' : 'log-los (std)'}
    
    df.rename(columns = rename, inplace=True)
    
    scaler = MinMaxScaler()
    
    df_dual = pd.DataFrame(data = scaler.fit_transform(df),
                           columns = df.columns,
                           index = df.index)
    
    heat0 = sns.heatmap(df, annot=True, linewidths= 1, fmt='.3f') 
                 
    fig, ax = plt.subplots(figsize=(12, 5))

    color_map = plt.get_cmap('YlGnBu')
    color_map = ListedColormap(color_map(np.linspace(0.1, 0.75, 256)))
    
    heat = sns.heatmap(df_dual, annot=True, linewidths= 1,
                       cmap= color_map, ax = ax, fmt='.3f')
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.1, 0.5, 1])
    colorbar.set_ticklabels(['Low', 'Middle', 'High'])
    
    for t in range(len(heat0.texts)):
        txt = heat0.texts[t].get_text()
        heat.texts[t].set_text(txt)
    

    
    ax.set_title('SelfTraining Table (5-fold cross validation)')
    fig.savefig(os.path.join('results', 'SelfTraining_table.png'),
                dpi = 1200,)
    
    
    

def best_model_results(model_name = 'pca_over_SVC_linear_10.0'):
    
    path = os.path.join(mccv_path, model_name)
    
    probability_heatmap(pd.read_csv(os.path.join(path, 'probability.csv')),
                        model_name)
    
    cross_heatmap(pd.read_csv(os.path.join(path, 'cross_matrix.csv')),
                  model_name)
    
    detailed_score_heatmap(pd.read_csv(os.path.join(path, 'detailed_score.csv')),
                           model_name)
    
def probability_heatmap(df, name):
    
    names, classes = classes_names()
    
    w = df.mean(axis=0).values.reshape(classes, classes)  #ndarray
    w = np.around(w, decimals=3)
    
    prob_frame = pd.DataFrame(data = w, columns = names, index = names)	
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(prob_frame, annot=True, linewidths= 1, cmap="YlGnBu", ax = ax)
    ax.set_title('True class v.s. Predicted Probability Class')
    fig.savefig(os.path.join('results', name + '_probability.png'),
                dpi = 1200,
                bbox_inches = "tight")
    
def cross_heatmap(df, name):
    
    names, classes = classes_names()
    
    w = df.mean(axis=0).values.reshape(classes, classes)  #ndarray
    
    for i in range(classes):
        w[i] /= np.sum(w[i])
        
    w = np.around(w, decimals=3)
        
    cross_frame = pd.DataFrame(data = w, columns = names, index = names)	
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(cross_frame, annot=True, linewidths= 1, cmap="YlGnBu", ax = ax)
    ax.set_title('True class v.s. Predicted Class (mean)')
    
    fig.savefig(os.path.join('results', name + '_cross_prediction.png'),
                dpi = 1200,
                bbox_inches = "tight")

def detailed_score_heatmap(df, name):
  
    names, classes = classes_names()
    
    w = df.mean(axis=0).values.reshape(classes, 4)
    
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

    fig.savefig(os.path.join('results', name + '_detailed_score.png'),
                dpi = 1200,
                bbox_inches = "tight")
    
    

def final_table():
        
    names, classes = classes_names()
        
    ked_et_al = {'Cellulose acetate': 0.97,    
                     'Cellulose like': 0.65, 
                     'Ethylene propylene rubber': 0.76, 
                     'Morphotype 1': 0.89, 
                     'Morphotype 2': 0.88, 
                     'PEVA': 0.74, 
                     'Poly(amide)': 1, 
                     'Poly(ethylene)' : 1, 
                     'Poly(ethylene) + fouling' : 0.88,
                     'Poly(ethylene) like' : 0.69, 
                     'Poly(propylene)' : 0.99, 
                     'Poly(propylene) like' : 0.51, 
                     'Poly(styrene)' : 0.99,  
                     'Unknown' : 0 }
    w0 = []
        
    for n in names:
        w0.append(ked_et_al[n])
            
    w0 = np.array(w0)    
        
    #Load model's sensitivity mccv data (using Kedzierski et. al methodology)
    df1 = pd.read_csv(os.path.join('results', 
                                  'final_model_mccv_all_data_detailed_score.csv'),
                      index_col=0)
                                       
    w1 = df1.mean(axis=0).values.reshape(classes, 4)
    w1 = np.around(w1, decimals=3)[:, 0]
    
    #Load MCCV results (best model)
    df2 = pd.read_csv(best_path) 
    w2 = df2.mean(axis=0).values.reshape(classes, 4)
    w2 = np.around(w2, decimals=3)[:, 0] 
        
    #Load model's sensitivity test result
    df3 = pd.read_csv(os.path.join('results','final_test_detailed_score.csv'))
    w3 = df3.mean(axis=0).values.reshape(classes, 4)
    w3 = np.around(w3, decimals=3)[:, 0] 
    
    
    w = np.stack((w0, w1, w2, w3), axis=0)
    
    df = pd.DataFrame(data = w,
                        columns= names,
                        index = ["Kedzierski et al.", 
                                 "SVC + Kedzierski et al", 
                                 "SVC + MCCV",
                                 "SVC Final test"])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('Sensitivity comparison')

    sns.heatmap(df,
                annot=True, linewidths= 0.05, cmap='YlGnBu', ax = ax)
    
    fig.savefig(os.path.join('results', 'sensitivity_final_table.png'),
                dpi = 1200,
                bbox_inches = "tight")

        
    return df

def mccv_files():
    return [model for model in os.listdir(mccv_path) if 
            os.path.isdir(os.path.join(mccv_path, model))  ]

        