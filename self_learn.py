#Standard 
import os
import pandas as pd
import numpy as np
from timeit import default_timer as timer

#Graphics
from matplotlib import pyplot as plt

#Sklearn modules
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
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

dic = {
       'SVC_linear': 
           [SVC(kernel = 'linear', C = 10, probability = True, 
                random_state = seed),
            None,
            os.path.join('results', 'pca_over_gs.csv')],
        'LogisticRegression':  best_estimator('LogisticRegression'),
        'SVC': best_estimator('SVC'),
        'NN':  best_estimator('NN')
       }
    
def run_self_learn():
    
    print('Loading data...')
    X_unlabel = pd.read_csv(os.path.join('data', 'unlabeled_data.csv')) 
    X_unlabel.drop(['3997.91411'], axis = 1, inplace=True)
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 

    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel() 
    y_unlabel = -1*np.ones(X_unlabel.shape[0])
    
    t_values = np.arange(0.3, 1, 0.1)
    folds = StratifiedKFold(n_splits= n_folds, shuffle=False) 
    
    for name in dic:
        start = timer()
        model = dic[name][0]
        base_path = os.path.basename(dic[name][2])
        _, scaler, pca, over_sample = pipe_config(base_path)

        
        scores = np.empty((t_values.shape[0], n_folds))
        accuracy_scores = np.empty((t_values.shape[0], n_folds))
        amount_labeled = np.empty((t_values.shape[0], n_folds))
        amount_iterations = np.empty((t_values.shape[0], n_folds))

        for (i, threshold) in enumerate(t_values):    
        
            print("threshold: {}".format(threshold))
        
            for fold, (train_index, test_index) in enumerate(
                    folds.split(X_train, y_train)):
            
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
                amount_labeled[i, fold] = y_unlabel.shape[0] - np.unique(
                    semi_model.labeled_iter_, return_counts=True)[1][0]
            
                # The last iteration the classifier labeled a sample in
                amount_iterations[i, fold] = np.max(semi_model.labeled_iter_)
                
                scores[i, fold] = log_loss(y_test, 
                                           semi_model.predict_proba(X_test))
                
                accuracy_scores[i, fold] = accuracy_score(y_test, 
                                                          semi_model.predict(
                                                              X_test))
                
        end = timer() 
        
        time = np.round(end - start, 4)
        loss_list = np.round(np.mean(scores, axis = 1), 4)
        loss = np.min(loss_list)
        idx = np.argmin(loss_list)
        labaled = np.round(np.mean(amount_labeled, axis = 1), 4)[idx]
        accuracy = np.round(np.mean(accuracy_scores, axis = 1), 4)[idx]
        std_log_loss = np.round(np.std(scores, axis = 1, ddof = 1), 4)[idx]

        
        with open(os.path.join('results', 'SelfTraining.csv'), 'a+') as f_object:
            f_object.seek(0)
            data = f_object.read(100)
            if len(data) > 0 :
                f_object.write("\n")
            f_object.write(','.join([name,
                                     base_path,
                                     str(time),
                                     str(labaled),
                                     str(accuracy),
                                     str(loss),
                                     str(std_log_loss)]))
        
        ax1 = plt.subplot(211)
        ax1.set_title('{} Self Training'.format(name))        
        ax1.errorbar(t_values, scores.mean(axis=1),
                     yerr=scores.std(axis=1),
                     capsize=2, color='b')
        ax1.set_ylabel('Log-loss', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlabel('Threshold')
    
        ax2 = ax1.twinx()
        ax2.errorbar(t_values, amount_labeled.mean(axis=1),
                     yerr=amount_labeled.std(axis=1),
                     capsize=2, color='g')
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('Amount of labeled samples', color='g')
        ax2.tick_params('y', colors='g')
    
        fig = ax1.get_figure()
        plt.show()
    
        fig.savefig(os.path.join('results', '{}_SelfTraining.png'.format(name)), 
                    dpi = 1200,
                    bbox_inches = "tight")