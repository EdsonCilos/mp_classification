#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:46:31 2020

@author: edson
"""
import numpy as np
from scipy import stats

def shapiro_test(data, conf = 0.01):

  gaussian_features = []

  for col in data:

     _, p_value = stats.shapiro(data[col])

     if p_value > conf: 
       gaussian_features.append(col)
  
  return gaussian_features

def kolmogorov_test(data, conf = 0.01):

    gaussian_features = []

    for col in data:
      
      _, p_value = stats.kstest(data[col], 'norm', \
                            args=(np.mean(data[col]), np.std(data[col])))
      
      if p_value > conf: 
        gaussian_features.append(col)
        
    return gaussian_features
