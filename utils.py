#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:30:23 2020

@author: Edson Cilos
"""
import os
import numpy as np
import pickle
from sklearn.metrics import multilabel_confusion_matrix
    
def sensitivity(matrix):
  return matrix[1,1]/(matrix[1,1] + matrix[1,0])

def specificity(matrix):
 return matrix[0,0]/(matrix[0,0] + matrix[0,1])

def precision(matrix):
  a = matrix[1,1] + matrix[0,1]
  return matrix[1,1]/a if a != 0 else 0

def f1(matrix):
  _precision = precision(matrix)
  recall = sensitivity(matrix)
  return 2 * (_precision * recall) / (_precision + recall) \
if _precision + recall != 0 else 0

def array_result(multi_matrix, index):
  matrix = multi_matrix[index]
  return [sensitivity(matrix),
          specificity(matrix),
          precision(matrix),
          f1(matrix)
          ]

def build_row(X_test, y_test, y_pred):

    multi_matrix = multilabel_confusion_matrix(y_test, y_pred)
    
    result = []
    
    for i in range(np.unique(y_test).shape[0]):
        result.extend(array_result(multi_matrix, i))
  
    return result        

def file_name(nn = False, sv_filter=False, scaler=False, 
              pca=False, over_sample=False):
    
    prefix = 'nn_' if nn else ''
    sc = 'std_' if scaler else ''
    sv = 'svfilter_' if sv_filter else '' 
    pc = 'pca_' if pca else ''
    ov = 'over_' if over_sample else ''
    
    return  prefix + sv + sc +  pc + ov + 'gs.csv'

def load_encoder():
    return pickle.load(open(os.path.join('data', 'enconder.sav'), 'rb'))


def classes_names():
    encoder =load_encoder()
    classes = len(encoder.classes_)
    return encoder.inverse_transform([i for i in range(classes)]), classes