# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:37:51 2021

@author: Edson Cilos
"""
import os

seed = 0
scalers = ['std']

def _seed():
    return seed

def _scaler_list():
    return scalers

def _mccv_path():
    return os.path.join('results', 'mccv')