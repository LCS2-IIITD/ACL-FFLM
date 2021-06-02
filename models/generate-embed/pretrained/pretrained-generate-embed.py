import nltk
import jsonlines
import wget
import gensim
import torch
import numpy as np
import jsonlines
import datetime
import transformers
import os
import csv
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, TFBertModel,BertModel,RobertaTokenizer, TFRobertaModel,RobertaModel, XLNetTokenizer,TFXLNetModel,OpenAIGPTTokenizer,OpenAIGPTModel,GPT2Tokenizer,GPT2Model,TFOpenAIGPTModel,TFGPT2Model
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
import gensim
import sys
import re
import copy
import pathlib
from pathlib import Path


sys.path.insert(0,'../..')
from read_write_functions import *
# from make_datasets.make_dataset import *
sys.path.insert(0,'../../..')
from evaluation_metrics.precision_recall_metrics import *


DUMP_MODE = True

# Load Pretrained Models
def get_pretrained_model(model_name = 'doc2vec'):
    tokenizer = None
    model = None
    if model_name == 'doc2vec':
        print("Loading Doc2Vec Model........")
        path_to_ap_d2v_model = "/home/nirav17072/BTP/data/pretrained_models/doc2vec/apnews_dbow/doc2vec.bin"
        model = gensim.models.Doc2Vec.load(path_to_ap_d2v_model)
        print("Doc2Vec Model Loaded 100%")
    elif model_name == 'bert-base-uncased' or model_name == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = TFBertModel.from_pretrained(model_name)    
    elif model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = TFRobertaModel.from_pretrained(model_name)
    elif model_name == 'xlnet-base-cased':
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        model = TFXLNetModel.from_pretrained(model_name)
    elif model_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
        model = TFOpenAIGPTModel.from_pretrained(model_name)
        # Add Special Token CLS in GPT2 tokenizer 
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))

    elif model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = TFGPT2Model.from_pretrained(model_name)
        # Add Special Token CLS in GPT2 tokenizer
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer


# Genrate EMbeddings
def generate_embeddings(model_name,model,sentence,tokenizer = None,batch_size = 32):
    # To get the sentence level embedding of a sentence the process is - 
    # 1. Add a special token to the sequence
    # [CLS] - BERT, <s> - RoBERTa, <cls> - XLNet, [CLS] - GPT, GPT2
    # 2. Tokenize the sentence using the toknizer
    # For BERT, RoBERTa (at the beginning) and XLnet (at the end) are automatically added when add_special_tokens parameter isTrue
    # For GPT and GPT2, manually adding the [CLS] at the end of the sentence
    # https://huggingface.co/transformers/model_doc/gpt2.html,  
    # https://github.com/huggingface/transformers/issues/3168 
    # 3. The sentence level embedding of the sentence is the word representation of the token

    # Am currently sending one sentence at a time need to do in batches
    # This simplifies this things and does not require any padding
    
    short_name = model_name[:4] 
    if short_name == 'doc2':
        # Doc2vec
        embeddings = model.infer_vector(sentence)
        return embeddings.reshape(-1,1).T 
    
    elif short_name == 'bert' or short_name == 'robe':
        tokens = tokenizer.encode(sentence, add_special_tokens = True)
        
        # BERT and RoBERTa have a max token length of 512. 
        nftokens = len(tokens)
        if nftokens > 512:
            print('Exceeds, Number of tokens in the sentences = ', nftokens)
            tokens = tokens[:512]
        input_ids = tf.constant(tokens)[None, :]
        
        #gen_sent = tokenizer.decode(tokens)
        #print(gen_sent)
        
        # Source  - https://github.com/huggingface/transformers/issues/1950
        # Which embedding to pick 
        outputs = model(input_ids)
        embeddings_of_last_layer = outputs[0]
        # There is only 1 sentence in the batch 
        first_sentence_embeddings = embeddings_of_last_layer[0]
        embeddings = first_sentence_embeddings[0]
        
    elif short_name == 'xlne':
        input_ids = tf.constant(tokenizer.encode(sentence, add_special_tokens = True))[None, :]
        outputs = model(input_ids)
        embeddings_of_last_layer = outputs[0]
        embeddings = embeddings_of_last_layer[0][-1]
        
    elif short_name == 'open' or short_name == 'gpt2':  
        # For GPT and GPT2 models, tokenizers do not contain [CLS] token
        # Manaully add the [CLS] at the end of the sentence, add [CLS] token in tokenizer dictionary
        # Input IDs are the token ID's in the dictionary
        
        sentence += ' [CLS]'
        tokens = tokenizer.encode(sentence, add_special_tokens = True)
        
        cls_token_location = [tokens.index(tokenizer.cls_token_id)]
        
        #print('cls_token_location = ',cls_token_location)
        
        input_ids = tf.constant(tokens)[None, :]
        #gen_sent = tokenizer.decode(tokens)
        #print(gen_sent)
        # Getting the embeddings from the model
        outputs = model(input_ids)
        
        # Embeddings 
        embeddings_of_last_layer = outputs[0]
        
        # CLS token embeddings 
        embeddings = embeddings_of_last_layer[0][-1]
        
    return embeddings.numpy().reshape(-1,1).T 

def convert(lst, var_lst): 
    it = iter(lst) 
    return [list(islice(it, i)) for i in var_lst] 

import time
def gltr_process_file(class_name, comment_dictionary, output_filepath_arr, output_filepath_dict, limit ,thread_no):
    bert_lm = BERTLM()
    gpt2_lm = LM()
    start = time.time()
    
    for label in comment_dictionary:
        
        # Counters
        global_sent_no = 0
        class_comment_no = 0
        
        complete_label_set_arr = []
        complete_label_set_dict = {}
        
        #print(complete_label_set_dict[len(complete_label_set_dict) - 1])
        
        break_now = False
        print('-------------------- Class Name = ',label,thread_no,'---------------------------------')
        
        for obj in comment_dictionary[label]:
            
            # Body or text of the sentence
            comment = obj[0]['body']
            
            # Tokenizing the sentence
            sentences = sent_tokenize(comment)
            
            # Sentence Number 
            local_sent_no = 0
            
            for sentence in sentences:
                #print('Generating GLTR features .......',model_name, thread_no, label)
                
                if global_sent_no % 50 == 0:
                    bert_avg_features, bert_bucket_vals, bert_token_wise_features = make_features(bert_lm,gpt2_lm,sentence, 'bert', top_k = 2, verbose = True)
                    gpt2_avg_features, gpt2_bucket_vals, gpt2_token_wise_features = make_features(bert_lm,gpt2_lm,sentence, 'gpt2', top_k = 2,verbose = True) 
                else:
                    bert_avg_features, bert_bucket_vals, bert_token_wise_features = make_features(bert_lm,gpt2_lm,sentence, 'bert', top_k = 2)
                    gpt2_avg_features, gpt2_bucket_vals, gpt2_token_wise_features = make_features(bert_lm,gpt2_lm,sentence, 'gpt2', top_k = 2) 
                    
                
                # Storing the label(class name), global_sent_no,com no, local sent no, sentence, sentence embeddings in an array
                sample =[label, global_sent_no, class_comment_no, local_sent_no, sentence,
                         bert_avg_features, gpt2_avg_features, bert_token_wise_features, 
                         gpt2_token_wise_features, bert_bucket_vals, gpt2_bucket_vals, False]
                
                print('old sample')
                #print(complete_label_set_dict[len(complete_label_set_dict) - 1])
                
                #print('Adding the to the label set .....')
                complete_label_set_dict[global_sent_no] = {
                            'global_sent_no' : global_sent_no,
                            'class_comment_no': class_comment_no,
                            'local_sent_no': local_sent_no,
                            'sent': sentence,
                            'gltr': {
                                'bert':{
                                    'avg_features': bert_avg_features,
                                    'bucket_vals': bert_bucket_vals,
                                    'token_wise_features': bert_token_wise_features
                                },
                                'gpt2':{
                                    'avg_features': gpt2_avg_features,
                                    'bucket_values': gpt2_bucket_vals,
                                    'token_wise_features': gpt2_token_wise_features
                                }
                            },
                            'complete':False
                        }
                
                complete_label_set_arr.append(sample)
                
                print('New Sample = ')
                print(complete_label_set_dict[global_sent_no])
                
                
                
                if global_sent_no % 100 == 0:
                    print('Dumping after every 10 samples embeddings.....') 
                    dumpPickleFile(complete_label_set_arr, output_filepath_arr)
                    dumpPickleFile(complete_label_set_dict, output_filepath_dict)
                    print(class_name,'-----------',global_sent_no)
                    current = time.time()
                    print('Time till now = ',current - start,'seconds')
                    
                #print('Updating the counters....') 
                
                if global_sent_no == limit:
                    print('Done', limit, 'samples Stopping.....') 
                    dumpPickleFile(complete_label_set_arr, output_filepath_arr)
                    dumpPickleFile(complete_label_set_dict, output_filepath_dict)
                    print(class_name,'-----------',global_sent_no)
                    break_now = True
                    break
                
                global_sent_no += 1
                local_sent_no += 1
            
            if break_now == True:
                break
            class_comment_no += 1
        
        # Final dump
#         complete_label_set_arr[-1][-1] = True
#         complete_label_set_dict[global_sent_no]['complete'] = True
        
        print('Dumping Pickle File.....')
        dumpPickleFile(complete_label_set_arr, output_filepath_arr)
        dumpPickleFile(complete_label_set_dict, output_filepath_dict)
        
        print('Completed - ',sample[:5], class_name, thread_no, label)
        print('Number of Sentences = ', global_sent_no)

        
def process_thread(model_name, full_dictionary, commentIDs, thread_no):
    
    
    print('------------------------',thread_no,'-------------------')
    
    model, tokenizer = get_pretrained_model(model_name = model_name)
    
    all_embeddings = []
    all_labels = []
    
    for comment_no, commentID in enumerate(commentIDs):
        class_no = commentID.split('-')[0][1:]
        x = full_dictionary[class_no]['data'][commentID]
        comment = x['comment']
        classname = x['classname']
        
        comment_embeddings = generate_embeddings(model_name, model, comment, tokenizer)
        comment_embeddings = comment_embeddings.reshape(1,-1)
        
        if comment_no % 1000 == 0:
            print(comment_no, thread_no)
        
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
synthetic = True

# Fold Number 1
fold_number = 1

model_name = 'roberta-base'
nf_train_samples = 800
nf_val_samples = 100
nf_test_samples = 200
num_classes = 108
min_nf_tokens = 6
data_type = 'train'
num_threads = 1

DUMP_MODE = True

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

#----------------Parameters-------------------------------------------------------------------------

# Not using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
output_folder = working_folder + 'results/generate-embed/pretrained/' + model_name + '/' + str(min_nf_tokens) +  '/' + path_str + '/'
create_folders(output_folder)

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
    


dumpPickleFile([combined_embeddings, combined_labels], output_folder + data_type + '_embeddings.pkl')

    