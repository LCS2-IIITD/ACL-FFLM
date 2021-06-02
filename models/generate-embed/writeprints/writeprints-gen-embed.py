import nltk
import jsonlines
import wget
import gensim
# import torch
import numpy as np
import jsonlines
import datetime
# import transformers
import os
import csv
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
# from transformers import BertTokenizer, TFBertModel,BertModel,RobertaTokenizer, TFRobertaModel,RobertaModel, XLNetTokenizer,TFXLNetModel,OpenAIGPTTokenizer,OpenAIGPTModel,GPT2Tokenizer,GPT2Model,TFOpenAIGPTModel,TFGPT2Model
from itertools import islice
nltk.download('punkt')
import pickle
import json
import random
from sklearn.utils import shuffle
import pathlib
from threading import Thread
from time import sleep
from multiprocessing import Process
import multiprocessing 
import gensim
import sys
import re
import copy
import pathlib
from pathlib import Path
#from writeprints.writeprints.text_processor import Processor
# Flatten will split vectorized featurs into individual featurs


sys.path.insert(0,'../..')
from read_write_functions import *
# from make_datasets.make_dataset import *
# from evaluation_metrics.precision_recall_metrics import *

sys.path.insert(0,'.')
from writeprints1.writeprints.text_processor import Processor

DUMP_MODE = True


def generate_embeddings(comment, processor):
    features = processor.extract(comment)
    arr = []
    for feature_name in features:
        #print(feature_name)
        x = features[feature_name]
        if type(x) == list:
            #print(len(x))
            arr += features[feature_name]
        else:
            arr.append(x)
    
    assert(len(arr) == 220)
    return np.array(arr)

def convert(lst, var_lst): 
    it = iter(lst) 
    return [list(islice(it, i)) for i in var_lst] 
        
def process_thread(model_name, full_dictionary, commentIDs, thread_no):
    
    print('------------------------',thread_no,'-------------------')
    processor = Processor(flatten = False) 
    
    all_embeddings = []
    all_labels = []
    
    for comment_no, commentID in enumerate(commentIDs):
        class_no = commentID.split('-')[0][1:]
        x = full_dictionary[class_no]['data'][commentID]
        comment = x['comment']
        classname = x['classname']
        
        comment_embeddings = generate_embeddings(comment, processor)
        comment_embeddings = comment_embeddings.reshape(1,-1)
        
        if comment_no % 1000 == 0:
            print(comment_no, thread_no,comment,'---------')
            #print(comment_embeddings)
        
        if comment_no == 0:
            all_embeddings = comment_embeddings
        else:
            all_embeddings = np.concatenate((all_embeddings,comment_embeddings), axis = 0)
        
        all_labels.append(int(class_no))
        
    return_dictionary[thread_no] =  [all_embeddings, all_labels]

    
def get_all_datasets(dataset_list_path,data_folder):
    all_datasets = []
    datasets = []
    d = openCSVfile(dataset_list_path, delimiter = ",")
    for i in d:
        all_datasets.append(i[0])
    c = 0
    for filename in all_datasets:
        print("------------",filename,"-----------")
        filepath = data_folder + filename + '.txt'
        if os.path.exists(filepath):
            c += 1
            datasets.append(filename)
        else:
            print(filename,'filepath does not exist')
            continue
    return datasets

    
def create_folders(folder_path):
    path = Path(folder_path)
    path.mkdir(parents = True,exist_ok = True)

#---------------------------------------------------

# Is it a balanced Dataset
balanced = True
# Lowercase the sentence for embedding generation
lowercase_sentence = False

# Fold Number 1
fold_number = 1

model_name = 'writeprints'
nf_train_samples = 800
nf_val_samples = 100
nf_test_samples = 200
num_classes = 108
min_nf_tokens = 6
data_type = 'train'
num_threads = 15

DUMP_MODE = True
synthetic = True


loss_type = 'cross_entropy'
# loss_type = 'multilabel'

one_hot = False

if loss_type == 'multilabel': 
    one_hot = True

#------------------------------------------PATHS----------------------------------------------------------
path_str = str(num_classes) + '_' + str(nf_train_samples) + '_' + str(nf_val_samples) + '_' + str(nf_test_samples)
working_folder = "../../../"

#---------Input Folder-----------
# Filename - 108_800_100_200_dataset.json
input_ids_path = working_folder + 'dataset/'

if synthetic == True:
    input_ids_path += 'synthetic'
else:
    input_ids_path += 'organic'

input_ids_path += '/splits/' + str(min_nf_tokens) + '/' + path_str + '_dataset.json'
complete_dataset_path = working_folder + 'dataset/'

if synthetic == True:
    complete_dataset_path += 'synthetic'
else:
    complete_dataset_path += 'organic'
    
complete_dataset_path +=  '/dataset.json'

# Not using GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
output_folder = working_folder + 'results/generate-embed/writeprints/'  + str(min_nf_tokens) +  '/' + path_str + '/'

# All available models
pretrained_model_names = ["roberta-base"]

# Class Names
authors_info = openCSVfile(working_folder + 'data/authors.csv',delimiter = ',')
class_names = [author_info[0] for author_info in authors_info][:num_classes]


# ID's 
ids = loadJsonFile(input_ids_path)
complete_dataset = loadJsonFile(complete_dataset_path)


processes = []
process_no = 0

manager = multiprocessing.Manager()
return_dictionary = manager.dict()


# Dividing threads
parts = np.array_split(ids[data_type], num_threads)

# 
for process_no, part in enumerate(parts):
    batch_comment_IDs = list(part)
    c_dataset = copy.deepcopy(complete_dataset)
    print('Making process ....', process_no)
    process = Process(target = process_thread, args = (model_name, c_dataset , batch_comment_IDs, process_no))
    processes.append(process)
    
# Starting process
for process in processes:
    process.start()

# Ending process
for process in processes:
    process.join()

combined_embeddings = []
combined_labels = []

for process_no in range(num_threads):
    [all_embeddings, all_labels] = return_dictionary[process_no]
    
    print(all_embeddings.shape, len(all_labels))
    
    if process_no == 0:
        combined_embeddings = all_embeddings
        combined_labels = all_labels
    else:
        combined_embeddings = np.concatenate((combined_embeddings, all_embeddings), axis = 0)
        combined_labels += all_labels
        
    print(combined_embeddings.shape, len(combined_labels))

create_folders(output_folder)
dumpPickleFile([combined_embeddings, combined_labels], output_folder + data_type + '_embeddings.pkl')

    