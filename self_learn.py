#Standard 
import os
import pandas as pd
import numpy as np

#Graphics
from matplotlib import pyplot as plt

#Sklearn modules
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

#Imblearn modules
from imblearn.over_sampling import RandomOverSampler

#Project modules
from model_picker import best_estimator
from pipeline import pipe_config
import config

#Fix seed to reproducibility
seed = config._seed() 

n_folds = 5

#here we use thee seed in best_estimator
#If you wish to override the seed, please consider here
"""
name = 'LogisticRegression'
model, _, path = best_estimator(name) 
"""
name = 'SVC_linear'
model = SVC(kernel = 'linear', C = 10, probability = True, random_state = seed)
path = os.path.join('results', 'pca_over_gs.csv')
_, scaler, pca, over_sample = pipe_config(os.path.basename(path))



t_values = np.arange(0.9, 1, 0.02)
folds = StratifiedKFold(n_splits= n_folds, shuffle=False) 
scores = np.empty((t_values.shape[0], n_folds))
amount_labeled = np.empty((t_values.shape[0], n_folds))
amount_iterations = np.empty((t_values.shape[0], n_folds))


print('Loading data...')

X_unlabel = pd.read_csv(os.path.join('data', 'unlabeled_data.csv')) 
X_unlabel.drop(['3997.91411'], axis = 1, inplace=True)
X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 

y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel() 
y_unlabel = -1*np.ones(X_unlabel.shape[0])

for (i, threshold) in enumerate(t_values):    
    
    print("threshold: {}".format(threshold))
    
    for fold, (train_index, test_index) in enumerate(folds.split(X_train, 
                                                                 y_train)):
        
        print("Fold {}/{}".format(fold + 1, n_folds))
            
        
        semi_model = SelfTrainingClassifier(clone(model), 
                                            threshold = threshold,
                                            verbose=True)
        
        
        X_train_temp = X_train.loc[train_index].copy()
        X_test =  X_train.loc[test_index].copy()
        
        
        y_train_temp = y_train[train_index].copy()
        y_test = y_train[test_index].copy()
        
        X_unlabel_processed = X_unlabel.copy()
        
        if(scaler):
            std = StandardScaler()
            X_train_temp = std.fit_transform(X_train_temp)
            X_test = std.transform(X_test)
            X_unlabel_processed = std.transform(X_unlabel_processed)
            
        if(pca):
            pca = PCA(n_components=0.99, random_state = seed)
            X_train_temp = pca.fit_transform(X_train_temp)
            X_test = pca.transform(X_test)
            X_unlabel_processed = pca.transform(X_unlabel_processed)
            
        if(over_sample):
            ros = RandomOverSampler(random_state = seed)
            X_train_temp, y_train_temp = ros.fit_resample(X_train_temp, 
                                                          y_train_temp)
            
        X_semi = pd.concat([pd.DataFrame(X_train_temp),
                            pd.DataFrame(X_unlabel_processed)],
                           axis=0,
                           ignore_index=True)
        
        y_semi = np.concatenate((y_train_temp, y_unlabel))
                
        semi_model.fit(X_semi, y_semi)
        
         
        # The amount of labeled samples that at the end of fitting
        amount_labeled[i, fold] = y_semi.shape[0] - np.unique(
            semi_model.labeled_iter_, return_counts=True)[1][0]
        
        # The last iteration the classifier labeled a sample in
        amount_iterations[i, fold] = np.max(semi_model.labeled_iter_)
        scores[i, fold] = log_loss(y_test, semi_model.predict_proba(X_test))
        
fig, ax = plt.subplots(figsize=(5, 7))
ax.set_title('Self Training')        

ax1 = plt.subplot(211)
ax1.errorbar(t_values, scores.mean(axis=1),
             yerr=scores.std(axis=1),
             capsize=2, color='b')
ax1.set_ylabel('Log-loss', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.errorbar(t_values, amount_labeled.mean(axis=1),
             yerr=amount_labeled.std(axis=1),
             capsize=2, color='g')
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Amount of labeled samples', color='g')
ax2.tick_params('y', colors='g')

ax3 = plt.subplot(212, sharex=ax1)
ax3.errorbar(t_values, amount_iterations.mean(axis=1),
             yerr=amount_iterations.std(axis=1),
             capsize=2, color='b')
ax3.set_ylim(bottom=0)
ax3.set_ylabel('Amount of iterations')
ax3.set_xlabel('Threshold')

plt.show()

fig.savefig(os.path.join('results', '{}_self_training.png'.format(name)), 
                dpi = 1200,
                bbox_inches = "tight")