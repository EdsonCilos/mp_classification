# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 23:57:33 2021

@author: Edson Cilos
"""

#Standard Modules
import os
import glob
import pandas as pd


def check_in(values_list, name):
    for value in values_list:
        if value in name: 
            return False
    return True

def build_data():
    
    i=0

    m8 = os.path.join('data', 'IR_Spectra', '050614', 'M8') 
    m14 = os.path.join('data', 'IR_Spectra', '060614', 'M14') 
    m21 = os.path.join('data', 'IR_Spectra', '080614', 'M21') 
    m23 = os.path.join('data', 'IR_Spectra', '080614 ', 'M23') 
    m209 = os.path.join('data', 'IR_Spectra', '011014 ', 'M209') 

    exclude = [m8, m14, m21, m23, m209]


    #Just to get the columns (supposing that all data in the same format)
    especific = os.path.join('data', 'IR_Spectra', '300914', 'M204', '500', 
                             'TM0033D1.txt')

    columns = pd.read_table(especific, header = None, sep=r'\s+')[0]
    
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
    


    
