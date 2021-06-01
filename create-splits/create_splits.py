import nltk
import jsonlines
import wget
import torch
import numpy as np
import jsonlines
import datetime
import os
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import islice
import pickle
import json
import random
from sklearn.utils import shuffle
import pathlib
from threading import Thread
from time import sleep
from multiprocessing import Process
import sys
import re
import copy
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def make_directory_tree(pathname):
	"""
	Creates  hierarchical paths
	"""
	path = Path(pathname)
	path.mkdir(parents = True, exist_ok = True)


def loadJsonFile(filepath, verbose = True, print_dict = False):
	"""
	Load a json file 
	"""
	if verbose == True : print("Loading a dictionary to filepath",filepath,".........")
	dictionary = {}
	
	with open(filepath) as jsonFile:
		dictionary = json.load(jsonFile)
	
	if verbose == True : print("Loaded Successfully")
	if print_dict == True : print(json.dumps(dictionary,indent = 4))
	return dictionary

def loadPickleFile(filepath, verbose = True, print_obj = False):
	"""
	Loading a Pickle File
	"""
	if verbose == True : print("Loading the pickle file from",filepath,"...........")
	
	pickle_in = open(filepath,"rb")
	example_dict = pickle.load(pickle_in)

	if verbose == True : print("Loaded the pickle File")
	if print_obj == True : print(example_dict)
	
	return example_dict


def dumpJsonFile(dictionary, filepath, verbose = True, print_dict = False):
	"""
	Dump a json file
	"""
	if verbose == True : print("Dumping a dictionary to filepath",filepath,"...............")
	
	with open(filepath,"w+") as jsonFile:
		json.dump(dictionary, jsonFile, indent = 4, sort_keys = True)
	
	if print_dict == True : print(json.dumps(dictionary,indent = 4))
	if verbose == True : print("Dumped Successfully")



working_folder = '../'
synthetic_data_folder = working_folder + 'data/synthetic/'
organic_data_folder = working_folder + 'data/organic/'
#----------------------PARAMS------------------------------
DUMP_MODE = True
synthetic = True
nfauthors = 108
train_size = nf_train_samples = 800
val_size = nf_val_samples = 100
test_size = nf_test_samples= 200
min_nf_tokens_in_comment = 6
random_state = 42


input_folder = working_folder + 'dataset/'
if synthetic == True: input_folder += 'synthetic/'
else : input_folder += 'organic/'
    
dict_output_filepath = input_folder + 'dataset.json' 
arr_output_filepath = input_folder  + 'dataset.pkl'

dictionary = loadJsonFile(dict_output_filepath)
array = loadPickleFile(arr_output_filepath)

output_folder =  working_folder + 'dataset/'

if synthetic == True: output_folder += 'synthetic/'
else : output_folder += 'organic/'
    
output_folder += '/splits/' + str(min_nf_tokens_in_comment) + '/'
dataset_path = output_folder + '/'  + str(nfauthors) + '_' +  str(train_size) + '_' + str(val_size) + '_' + str(test_size) +  '_dataset.json'


min_nftokens = []

for i in dictionary:
    c = 0
    for comment in dictionary[i]['data']:
        if dictionary[i]['data'][comment]['nftokens'] >= min_nf_tokens_in_comment : c += 1      
    print(dictionary[i]['classname'], c)
    min_nftokens.append(c)


def make_sets(indexes, comment_ids):
    ids = []
    for index in indexes : ids.append(comment_ids[index])
    return ids

def confirm_no_overlap(list1, list2):
    """
    Gives an assertion error if there is an overlap between 2 lists
    """
    set1, set2 = set(list1), set(list2)
    assert(len(set1.intersection(set2)) == 0)
    
def shuffle_multiple_arrays(commanding_array, array2):
    """
    Shuffles multiple arrays 
    """
    assert(len(commanding_array)  == len(array2))
    
    # Get a list of indexes
    indexes = list(range(len(commanding_array)))
    
    # Randomly Sort Indexes
    np.random.RandomState(random_state).shuffle(indexes)
    
    new_arr1 =  []
    new_arr2 = []
    for i in indexes:
        new_arr1.append(commanding_array[i])
        new_arr2.append(array2[i])
        
    return new_arr1, new_arr2


# Makes the largest dataset of 800 train, 100 validation and 200 test samples
def make_datasets(dictionary, min_token_size = 6, nfclasses = 108, train_size = 800, val_size = 100, test_size = 200):

    all_ids = {'train' : [], 'val' : [], 'test' : [], 'train_labels' : [], 'val_labels' : [], 'test_labels' : []}
    
    classes = list(range(nfclasses))
    tokens = []
    
    for class_no1 in classes:
        class_no = str(class_no1)
        
        class_size = len(dictionary[class_no]['data'])
        eligible_comment_ids = []
        class_ids = []
        
        for commentID in dictionary[class_no]['data']:
            nftokens = dictionary[class_no]['data'][commentID]['nftokens']
            
            if nftokens >= min_token_size: 
                eligible_comment_ids.append(commentID)
                class_ids.append(str(class_no1))
                tokens.append(nftokens)
        
        assert(len(class_ids) == len(eligible_comment_ids))
        
        nf_eligible_comments = eligible_comment_ids
        final_indexes = list(range(len(nf_eligible_comments)))
        np.random.RandomState(random_state).shuffle(final_indexes)
        
        # Getting the indexes
        train_indexes = final_indexes[:train_size]
        val_indexes = final_indexes[train_size : train_size  + val_size]
        test_indexes  = final_indexes[train_size  + val_size : train_size  + val_size + test_size]
        
        # Getting the comment IDs
        train_comment_ids = make_sets(train_indexes, eligible_comment_ids)
        val_comment_ids = make_sets(val_indexes, eligible_comment_ids)
        test_comment_ids = make_sets(test_indexes, eligible_comment_ids)
        
        # Getting the  labels
        train_labels = make_sets(train_indexes, class_ids)
        val_labels = make_sets(val_indexes, class_ids)
        test_labels = make_sets(test_indexes, class_ids)
        
        # Confirms there is no overlap between train, test and validation sets
        confirm_no_overlap(train_comment_ids, val_comment_ids)
        confirm_no_overlap(test_comment_ids, val_comment_ids)
        confirm_no_overlap(train_comment_ids, test_comment_ids)
        
        all_ids['train'] += train_comment_ids
        all_ids['test'] += test_comment_ids
        all_ids['val'] += val_comment_ids
        
        all_ids['train_labels'] += train_labels
        all_ids['test_labels'] += test_labels
        all_ids['val_labels'] += val_labels
        
    print(len(all_ids['train']))
    print(len(all_ids['test']))
    print(len(all_ids['val']))
    
    all_ids['train'],all_ids['train_labels'] =  shuffle_multiple_arrays(all_ids['train'], all_ids['train_labels'])
    
    print('Minimum nf tokens = ',sum(tokens)/len(tokens))
    
    return all_ids



largest_splits = make_datasets(dictionary, min_token_size = min_nf_tokens_in_comment, nfclasses = nfauthors, 
                       train_size = 800, val_size = 100, test_size = 200)






def reorganize_dataset(new_splits, nfclass = 108, nfsample_per_class = 200, datatype =  'train'):
    """
    Reorganizes the dataset for changed parameters
    
    NOTE : For our smaller experiments, (reduced #classes, reduced training samples), 
    I sampled the smaller training sets from the largest training set (800 samples per class). 
    I also kept the same test and validation set regardless of the train set size. 
    If the number of classes in train set were < 108, then we reconfigured the test and validatio set to discard those classes.
    """
    dataset = new_splits[datatype]
    labels = new_splits[datatype + '_labels']
    nf_class_labels = [0] * nfclass
    
    new_ids = []
    new_labels = []
    
    for label_no, label in enumerate(labels):
        class_label = int(label)
        
        # Restricting the class number
        if class_label >= nfclass: 
            continue
        
        # Checking the number  of samples per class  
        if nf_class_labels[class_label] < nfsample_per_class:
            nf_class_labels[class_label] += 1
            new_ids.append(dataset[label_no])
            new_labels.append(label)
        else:
            continue

    return new_ids, new_labels



# Now sampling for smaller configurations, recommendation : Do not change test and val size 
train_ids, train_labels = reorganize_dataset(largest_splits, nfclass = nfauthors, nfsample_per_class = train_size, datatype =  'train')
val_ids, val_labels = reorganize_dataset(largest_splits, nfclass = nfauthors, nfsample_per_class = val_size, datatype =  'val')

test_ids, test_labels = reorganize_dataset(largest_splits, nfclass = nfauthors, nfsample_per_class = test_size, datatype =  'test')


splits = {}

splits['train'] = train_ids
splits['train_labels'] = train_labels

splits['test'] = test_ids
splits['test_labels'] = test_labels

splits['val'] = val_ids
splits['val_labels'] = val_labels


# Makes k splits of training set
def make_k_splits(train_dataset, train_labels, number_of_splits = 5):
    """
    Divides the train dataset into k number of splits in a stratified manner
    """
    assert(len(train_dataset) == len(train_labels))
    
    splits = {}
    for i in range(1, number_of_splits + 1):
        splits[str(i)] = {}
    
    train_dataset = np.array(train_dataset)
    train_labels = np.array(train_labels)
    
    print(len(train_dataset))
    
    split_machine = StratifiedKFold(n_splits = number_of_splits, shuffle=False)
    split_no_s = 1
    
    for train_index, test_index in split_machine.split(train_dataset, train_labels):
        
        split_no = str(split_no_s)
        
        print("TRAIN:", train_index, len(train_index), "TEST:", test_index, len(test_index))
        
        X_train, X_test = train_dataset[train_index], train_dataset[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        
        splits[split_no]['train'] = list(X_train)
        splits[split_no]['train_labels'] = list(y_train)
        
        splits[split_no]['val'] = list(X_test)
        splits[split_no]['val_labels'] = list(y_test)
        
        split_no_s += 1
        
    return splits


k_fold_splits = make_k_splits(splits['train'], splits['train_labels'], number_of_splits = 5)
splits['5_fold_splits'] =  k_fold_splits
make_directory_tree(output_folder)

if DUMP_MODE == True:
    dumpJsonFile(splits, dataset_path, verbose = False, print_dict = False)
    


