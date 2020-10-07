#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:30:23 2020

@author: edson
"""
import os
import numpy as np
import pickle
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter

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

def Remove_less_representative(dataset, remove_n):

  less_rep = dataset["label"].value_counts()

  assert less_rep.shape[0] > remove_n, \
  f"There are {less_rep.shape[0]} classes, \
  not possible remove {remove_n} of them"

  less_rep = less_rep[less_rep.shape[0] - remove_n:]

  remove_idxs = [i for i, row in dataset.iterrows() 
  if row["label"] in less_rep.index]

  new_dataset = dataset.drop(index=remove_idxs, axis=0, inplace=False)

  return new_dataset

def load_encoder():
    return pickle.load(open(os.path.join('data', 'enconder.sav'), 'rb'))


def classes_names():
    encoder =load_encoder()
    classes = len(encoder.classes_)
    return encoder.inverse_transform([i for i in range(classes)]), classes