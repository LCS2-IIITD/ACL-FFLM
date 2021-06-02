import sys
import os
import pathlib
from multiprocessing import Process
import multiprocessing
import random
import matplotlib.pyplot as plt
#----------------------------------------------------
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve,average_precision_score,classification_report
from sklearn import *
import torch
import time
from sklearn.model_selection import PredefinedSplit
#----------------------------------------------------
from itertools import cycle
import numpy as np
import pandas as pd
#----------------------------------------------------
sys.path.insert(0,'..')
from read_write_functions import *
from sklearn_classifiers import *
# from make_datasets.make_dataset import *
sys.path.insert(0,'../..')
from evaluation_metrics.precision_recall_metrics import *
from hypopt import GridSearch



def make_paths(folder, model_name, classifier_name, DR_tech, path_str, min_nf_tokens):
    
    results_path = folder + 'results/' + model_name + '/' + str(DR_tech) + '/' + str(min_nf_tokens) + '/' + classifier_name + '/' + path_str + '.json'
    models_path = folder + 'models/' + model_name + '/' + str(DR_tech) + '/' +  str(min_nf_tokens) + '/'  classifier_name + '/' + path_str + '.pkl'
    metadata_path = folder + 'metadata/' + model_name + '/' + str(DR_tech) + '/' + str(min_nf_tokens) + '/'  classifier_name + '/' +  path_str + '.json'
    graph1_path =  folder + 'graph1/' + model_name + '/' + str(DR_tech) + '/' +  str(min_nf_tokens) + '/'  classifier_name  + '/' +  path_str + '.png'
    graph2_path =  folder + 'graph2/' + model_name + '/' + str(DR_tech) + '/' +  str(min_nf_tokens) + '/' classifier_name + '/'+  path_str + '.png'
    probs_path =  folder + 'test_probs/' + model_name + '/' + str(DR_tech) + '/' + str(min_nf_tokens) + '/'  classifier_name + '/'+  path_str + '.pkl'

    
    create_folders_path(graph1_path)
    create_folders_path(graph2_path)
    create_folders_path(results_path)
    create_folders_path(models_path)
    create_folders_path(metadata_path)
    create_folders_path(probs_path)
    
    paths = {
        'result': results_path,
        'model': models_path,
        'metadata': metadata_path,
        'graph1': graph1_path,
        'graph2': graph2_path,
        'test_probs': probs_path
    }
    
    return paths


def train_and_eval(best_clf, X_train, y_train, X_test, y_test, classifier_name, paths, DR_tech = None):
    
    print('fitting.....')
    train_start = time.time()
    best_clf.fit(X_train, y_train)
    train_end = time.time()
    
    y_score, Y_test = make_precision_recall_curve(best_clf, X_train, y_train, X_test, y_test, num_classes, 
                                                  paths['graph1'],paths['graph2'], DUMP_MODE = DUMP_MODE, classifier_name = classifier_name)
    
    print('predicting.....')
    test_start = time.time()
    test_pred = best_clf.predict(X_test)
    test_end = time.time()
    
    metadata = {
        'classifier': classifier_name,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'DR_tech': DR_tech,
        'train_time' : (train_end - train_start),
        'test_time' : (test_end - test_start)
    }
    
    class_report = classification_report(y_test, test_pred, output_dict = True, digits = 4, target_names = class_names)
    
    dumpPickleFile(best_clf, paths['model'])
    dumpJsonFile(class_report, paths['result'])
    dumpJsonFile(metadata, paths['metadata'])
    dumpPickleFile([y_score, y_test, test_pred], paths['test_probs'])

    a1,a2,a3 = class_report['macro avg']['f1-score'], class_report['accuracy'], class_report['weighted avg']['f1-score']
    
    print('Macro Avg = ', a1)
    print('Accuracy = ', a2)
    print('Weighted Avg = ', a3)
    
    class_report = classification_report(test_labels, test_pred, output_dict = False, digits = 4,target_names = class_names)
    
    print(class_report)

if __name__ == '__main__':
    
    
    pretrained_model_names = ["roberta-base"]
    num_classes_arr = [108]
    nftrainsamples_per_class_arr = [800]
    nfvalsamples_per_class_arr = [100]
    nftestsamples_per_class_arr = [200]
    balanced = True
    epoch_no = 3
    min_nf_tokens = 6


    #-------------------------------------------------PATHS--------------------------------------------------------
    # Set PATHS and everything over here
    working_folder = '../../'


    # Folders to save everthing in 

    folder_name = 'writeprints'
    store_folder = working_folder + '/results/generate-embed/' + folder_name + '/'
    output_folder = working_folder + '/results/classifiers/' + folder_name + '/'

    DR_tech = None
    DUMP_MODE = True
    dimen_size = None
    if DR_tech != None : dimen_size = 16

    seed_no = 42
    seed_everything(seed_no)


    return_dict = {}

    for num_classes in num_classes_arr:
        for i,nftrainsamples_per_class in enumerate(nftrainsamples_per_class_arr):

            nftestsamples_per_class = nftestsamples_per_class_arr[i]
            nfvalsamples_per_class = nfvalsamples_per_class_arr[i]

            pre_model_names = pretrained_model_names

            authors_info = openCSVfile(working_folder + 'data/authors.csv',delimiter = ',')
            class_names = [author_info[0] for author_info in authors_info][:num_classes]
            
            path_str = str(num_classes) + '_' +  str(nftrainsamples_per_class) + '_' + str(nfvalsamples_per_class) + '_' + str(nftestsamples_per_class)
            
            t_path_str = str(num_classes) + '_' +  str(800) + '_' + str(nfvalsamples_per_class) + '_' + str(nftestsamples_per_class)
            
            grid_search_cv_folder = output_folder + '/' + str(min_nf_tokens) + '/' + path_str + '/grid_search/'
            
            for model_name in pre_model_names:
                
                train_dataset_path =  store_folder + '/' + str(min_nf_tokens) + '/' + path_str + '/' + 'train_embeddings'
                val_dataset_path = store_folder + '/' + str(min_nf_tokens) + '/' + t_path_str + '/' +   'val_embeddings'
                test_dataset_path = store_folder + '/' + str(min_nf_tokens) + '/' + t_path_str + '/' +   'test_embeddings'
                
                if DR_tech != None:
                    train_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)
                    test_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)
                    val_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)

                train_dataset_path +=  '.pkl'
                test_dataset_path +=  '.pkl'
                val_dataset_path +=  '.pkl'
                
                
                if DUMP_MODE == True:
                    create_folders_path(grid_search_cv_folder)

                if DR_tech != None:
                    train_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)
                    test_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)
                    val_dataset_path +=   '_' + DR_tech + '_' + str(dimen_size)


                print('Path String = ',path_str)

                train_features, train_labels, test_features, test_labels = None, None, None, None
                print(train_dataset_path, test_dataset_path)

                def load_and_do(path):
                    x = loadPickleFile(path)
                    features, labels = x[0],x[1]
                    features = features.reshape(-1,220)
                    features = np.nan_to_num(features)
                    labels = np.array(labels)
#                     labels = list(range(108)) + list(range(108))
#                     features = features[:216,:10]

                    return features,labels

                train_features, train_labels =  load_and_do(train_dataset_path)
                test_features, test_labels =  load_and_do(test_dataset_path)
                val_features, val_labels =  load_and_do(val_dataset_path)


                val_index = []
                train_index = []
                for i in train_labels:train_index.append(-1)
                for j in val_labels:val_index.append(0)

                ps = PredefinedSplit(test_fold = train_index + val_index)

#------------------------------------------------------------------------------------------------------------------


#                 print(train_features.shape,train_labels.shape,test_features.shape,test_labels.shape)
                classifiers, parameters = initialize_classifiers(num_classes, classifier_type = 'fast')

#                 from sklearn.preprocessing import StandardScaler, MinMaxScaler

#                 print('Preprocessing .....')

#                 scaler = MinMaxScaler(feature_range = (-3,3))

#                 train_features = scaler.fit_transform(train_features)
#                 test_features = scaler.fit_transform(test_features)
#                 val_features = scaler.fit_transform(val_features)


                for classifier_name in classifiers:
        
#                     if classifier_name == 'knn':
#                         continue
                    
                    print('Model Name = ', model_name)
                    print('Classifier_name = ', classifier_name)
                    print('Number of classes = ',len(class_names))
                    print('Number of Train samples per class = ',nftrainsamples_per_class)
                    print('Number of Val samples per class = ',nfvalsamples_per_class)
                    print('Number of Test samples per class = ',nftestsamples_per_class)
                    
                    paths  = make_paths(output_folder, model_name, classifier_name, DR_tech, path_str, min_nf_tokens)

                    print('------------',classifier_name,'----------------')

                    clf = classifiers[classifier_name]
                    param_grid = parameters[classifier_name]

                    # Grid-search all parameter combinations using a validation set.
                    grid_kn = GridSearchCV(estimator = clf, return_train_score = True, param_grid = param_grid, 
                                           scoring = 'f1_macro', verbose = 1, n_jobs = -1 ,  cv=ps)


                    X = np.concatenate((train_features, val_features), axis = 0)
                    y = np.concatenate((train_labels, val_labels), axis = 0)
                    
                    print(X.shape)

                    grid_kn.fit(X, y)


                    # Dumping the GridSearchCV object
                    grid_cv_path = grid_search_cv_folder + model_name + '/' + classifier_name + '/' 
                    if DR_tech !=  None : grid_cv_path += DR_tech + '/'
                    grid_cv_path += path_str + '.pkl'

                    print(grid_cv_path)

                    if DUMP_MODE == True:
                        create_folders_path(grid_cv_path)
                        dumpPickleFile(grid_kn, grid_cv_path)

                    # Getting the best parameters
                    best_params = grid_kn.best_params_

                    print('---------------------', classifier_name, model_name,'-----------------------------------')

                    print('Setting the best params for the classifier......')
                    best_clf = classifiers[classifier_name]
                    best_clf.set_params(**best_params)

                    print('best_params = ',best_params)

                    train_and_eval(best_clf, train_features, train_labels, test_features, test_labels, classifier_name, 
                                   paths, DR_tech)

                    print('==========================================================================================')
