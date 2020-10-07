#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:02:39 2020

@author: edson
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import t
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot as qq
from matplotlib.colors import ListedColormap
import seaborn as sns
from utils import classes_names



def qq_plot(data):
    return qq(data, line='s')

def gs_heatmap():
    
    names = ['gs', 'pca_gs', 'pca_over_gs', 'over_gs']
    
    
    """
    if(prefix is not None):
        if(len(prefix) > 0):
            prefix += '_'
    else:
        prefix = ''
    """
    
    models = {'SVC' : 'SVC', 
              'RandomForestClassifier' : 'RF', 
              'LogisticRegression' : 'LR', 
              'KNeighborsClassifier' : 'KNN', 
              'DecisionTreeClassifier' : 'DT',
              'GaussianNB' : 'GB'}
    
    
    results_path = os.path.join(os.getcwd(), 'results')
    
    df_list = {}
    
    for name in names:
        gs_path = os.path.join(results_path, name + '.csv')
        df_list[name] = pd.read_csv(gs_path)
    
        
    c_map = plt.get_cmap('YlGnBu_r')
    c_map = ListedColormap(c_map(np.linspace(0.5, 0.8, 256)))
    
    
    #fig = plt.figure()

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    
    i = 0
    
    for name in df_list:
        
        ax = axs[i]
        i += 1
        
        
        df = df_list[name]
        
        rows = []
        idxs = []
    
        for key in models:
        
            row = next(r  for _, r  in df.iterrows() if key in r["estimator"])
        
            idxs.append(models[key])
        
            rows.append([-row["neg_log_loss"], row["std"]])
        
    
        df_ = pd.DataFrame(data = rows, 
                           columns = ["Log-loss", "Standard Deviation"],
                           index = idxs)
    
        df_.sort_values(by = ["Log-loss"], inplace = True)
        
        draw_bar = True if i == 4 else False

        sns.heatmap(df_, annot=True, linewidths= 1, cbar=draw_bar,
                    cmap=c_map, ax = ax, fmt='.4f')
        
        ax.set_title('Best model: ' + name )
    
    fig.savefig(os.path.join(results_path, 'gs_log_loss.png'),
                bbox_inches = "tight")
    
    return df_list
   
        

def log_loss_table():
    
    columns = ['Log-loss (avarage)', 'Standard deviation']
    
    algorithms = ['SVC_linear_0.001',
             'SVC_linear_0.0015',
             'SVC_linear_0.002',
             'SVC_linear_0.0025',
             'SVC_rbf_1.0', 
             'SVC_rbf_1.25',
             'SVC_rbf_1.44',
             'RF_100',
             'RF_500',
             'RF_1000',
             ]
    

    rows = []
    
    for name in algorithms:
        rows.append(log_loss_row(name))
        
    
        
    df = pd.DataFrame(data = np.round(rows, 4), 
                      columns = columns, 
                      index = algorithms)
    
    df.sort_values(by = 'Log-loss (avarage)', ascending=True, inplace = True)
    
        
    c_map = plt.get_cmap('YlGnBu')
    c_map = ListedColormap(c_map(np.linspace(0.1, 0.7, 256)))

    fig, ax = plt.subplots(figsize=(5, 7))
    sns.heatmap(df, annot=True, linewidths= 1, cmap=c_map, ax = ax, fmt='.4f')
    ax.set_title('Cross entropy')
    
    fig.savefig(os.path.join('results', 'log_loss.png'), 
                bbox_inches = "tight")
    
    return df
    
        

def log_loss_row(name, alpha = 1e-4):
  
    df = pd.read_csv(os.path.join('mhv_data', name,'total_score.csv'))

    return [np.mean(df[df.columns[1]]), np.std(df[df.columns[1]])]


def total_score_plot_all(alpha = 1e-4):
    
    _total_score_plot(['SVC_linear_0.001',
             'SVC_linear_0.0015',
             'SVC_linear_0.002',
             'SVC_linear_0.0025'
             ],'SVC_linear', alpha)
    
    _total_score_plot(['SVC_rbf_1.0',
             'SVC_rbf_1.25',
             'SVC_rbf_1.44'
             ],'SVC_rbf', alpha)
    
    _total_score_plot(['RF_100',
             'RF_500',
             'RF_1000'
             ],'RF', alpha)
    
    
def _total_score_plot(name_list, main_name, alpha):
    
    df_tuples = []

    for name in name_list:
        
        df = pd.read_csv(os.path.join('mhv_data', name,'total_score.csv'))
        
        n = df.shape[0]
        
        c = t.ppf(1 - alpha/2, n - 1, loc=0, scale=1)
        std = np.std(df[df.columns[1]])
        mean = np.mean(df[df.columns[1]])
        adjust = c*std/np.sqrt(n)
        
        #label1 = name
        label1 = name + ' loss: ' + str(round(mean, 5)) + \
            ' ±' +  str(round(adjust, 5))
            
        std = np.std(df[df.columns[3]])
        mean = np.mean(df[df.columns[3]])
        adjust = c*std/np.sqrt(n)
        
        #label2 = name
        label2 = name + ' score: ' + str(round(mean, 5)) + \
            ' ±' +  str(round(adjust, 5))
        
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
                bbox_inches = "tight")
    
    plt.show()
    plt.close()
    

def best_model_results(model_name = 'SVC_linear_0.002'):
    
    path = os.path.join('mhv_data', model_name)
    
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
                bbox_inches = "tight")
    