# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:50:53 2021

@author: Edson cilos
"""

import numpy as np

#Tensorflow API
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping


#Statitcs
from scipy.stats import reciprocal



#Fix seed to reproducibility
seed = 0

#Be aware about Keras's issue https://github.com/keras-team/keras/issues/13586
#Solution here https://stackoverflow.com/questions/62801440/kerasregressor-cannot-clone-object-no-idea-why-this-error-is-being-thrown/66771774#66771774



#Basic setup to build model
def build_model(n_hidden=1, 
                n_neurons=50, 
                momentum = 0.9,
                learning_rate=0.001, 
                act = "sigmoid"):
    
    model = keras.models.Sequential()
    
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation= act))
        
    model.add(keras.layers.Dense(14, activation="softmax"))
    
    optimizer = keras.optimizers.SGD(momentum = momentum,
                                     nesterov = True, 
                                     lr=learning_rate)
    
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    return model


def neural_grid(epochs = 1000, patience = 3):

    #Create dictionary with base model, including parameter grid
    neural_network = {'estimator': [KerasClassifier(build_model, 
                                 epochs = epochs,
                                 callbacks = [EarlyStopping(monitor='loss', 
                                                            patience= patience,
                                                            min_delta=0.001
                                                            )]
                                 )],
        "estimator__n_hidden": [1, 2, 3, 4, 5],
        "estimator__n_neurons": [10, 50, 100, 150, 200],
        "estimator__momentum" : np.arange(0.1, 0.9, 0.3),
        "estimator__learning_rate": [1e-3, 1e-2, 1e-1],
        "estimator__act": ["relu", "sigmoid", "tanh"],
        }
    
    return [neural_network]

  
"""
def shallow_sigmoid_search(pca = True, over_sample = True):
    
    #Load data
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel()
    
    #Base model
    base_model = KerasClassifier(build_model, 
                                 epochs = 400,
                                 callbacks = [EarlyStopping(monitor='loss', 
                                                            min_delta=0.005
                                                            patience=3)]
                                 )

    #Load pipeline
    pipe, file_name = build_pipe(pca, over_sample)
    
    #Create dictionary with base model, including parameter grid
    neural_network = base_dic(base_model)
    
    param_distribs = {"estimator__n_neurons": np.arange(45, 55)}
    
    neural_network.update(param_distribs)
    
    param_grid = [neural_network]

    cv_fixed_seed = StratifiedKFold(n_splits=4, 
                                    shuffle = True,
                                    random_state = seed)

    rnd_search = RandomizedSearchCV(pipe, 
                                    param_grid, 
                                    n_iter = 10,
                                    scoring = 'neg_log_loss',
                                    cv = cv_fixed_seed,
                                    n_jobs = 1,
                                    verbose = 10,
                                    random_state= seed)

    rnd_search.fit(X_train, y_train)

    results = pd.concat([pd.DataFrame(rnd_search.cv_results_["params"]),
                     pd.DataFrame(rnd_search.cv_results_['std_test_score'], 
                                  columns=["std"]),
                     pd.DataFrame(rnd_search.cv_results_["mean_test_score"], 
                                  columns=["neg_log_loss"])],axis=1)
    
    results.sort_values(by=['neg_log_loss'], ascending=False, inplace=True)
    
    file_name += 'shallow_sigmoid_ns.csv'
    
    folder = os.path.join(os.getcwd(), 'results')
    
    if not os.path.exists(folder):
            os.makedirs(folder)

    results.to_csv(os.path.join(folder, file_name), index = False)
    
    return results
"""