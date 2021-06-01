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


from nltk.tokenize import word_tokenize

def tokenize_words(words):
	"""
	Tokenize the words by nltk word tokenizers
	"""
	tokens = word_tokenize(words)
	return tokens


# sys.path.insert(0,'..')
# from read_write_functions import *
# from sklearn_classifiers import *
# from gltr.api import *

def dumpPickleFile(data, filepath, verbose = True, print_obj = False):
	"""
	Dumping a pickle file
	"""
	pickle_out = open(filepath,"wb")
	
	if verbose == True : print("Dumping the Pickle file into ",filepath,"...........")
	
	pickle.dump(data, pickle_out)
	
	if verbose == True : print("Dumped the pickle File")
	if print_obj == True : print(data)

	pickle_out.close() 


def openCSVfile(filepath, delimiter = ","):
    """
    """
    with open(filepath,"r") as csvfile:
        rows =  csv.reader(csvfile,delimiter = delimiter)
        return list(rows)

def dumpJsonFile(dictionary, filepath, verbose = True, print_dict = False):
	"""
	Dump a json file
	"""
	if verbose == True : print("Dumping a dictionary to filepath",filepath,"...............")
	
	with open(filepath,"w+") as jsonFile:
		json.dump(dictionary, jsonFile, indent = 4, sort_keys = True)
	
	if print_dict == True : print(json.dumps(dictionary,indent = 4))
	if verbose == True : print("Dumped Successfully")

""""PARAMETERS"""

working_folder = '../'
synthetic_data_folder = working_folder + 'data/synthetic/'
organic_data_folder = working_folder + 'data/organic/'
authors_info = openCSVfile(working_folder + 'data/authors.csv', delimiter = ',')

DUMP_MODE = True 
synthetic = True
nfauthors = 108

output_folder = working_folder + 'dataset/'

if synthetic == True:
    output_folder += 'synthetic/'
else:
    output_folder += 'organic/'
    
dict_output_filepath = output_folder + 'dataset.json' 
arr_output_filepath = output_folder  + 'dataset.pkl'

input_data_folder = synthetic_data_folder
if synthetic == False : 
    input_data_folder = organic_data_folder

authors = [author_info[0] for author_info in authors_info][:nfauthors]

def convert_timestamp(timestamp):
    """
    Convert  timestamp from reddit to day, month and year
    """
    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    date_split = date.split("-");month = int(date_split[1]);year = int(date_split[0])
    return month,year,date


def read_json_objects(filepath):
    """
    Read a jsonlines object
    """
    objects = [] 
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            #print(obj[0])
            #created_timestamp = obj['created_utc']#;updated_timestamp = obj['updated_utc']
            #month,year,date = convert_timestamp(created_timestamp)
            objects.append(obj)
    return objects

def create_unique_ID(class_no, sentence_no):
    """ 
    Creates a a unique ID for our dataset in the format of 'c' + class_no + '-s' + sentence_no. 
    Eg - c0-s0, c1-s2 etc.
    """
    unique_id = 'c' + str(class_no) + '-s' + str(sentence_no)
    return unique_id


def replace_link(comment):
    """" Replaces all links in the comment with a standard LINK tag"""
    copy_comment = copy.deepcopy(comment)
    tagged_comment = re.sub(r'http\S+', ' LINK ', copy_comment)
    return tagged_comment

def get_nf_tokens(comment):
    """Returns the number of tokens in the comment"""
    words = tokenize_words(comment)
    return len(words)

def preprocess(comment):
    """Pre-Process the comment"""
    copy_comment = copy.deepcopy(comment)
    
    # Replacing link
    final_comment = replace_link(copy_comment)
    nftokens = get_nf_tokens(comment)
    
    return final_comment, nftokens

def extract_data(final_comment, obj, unique_id , class_no, class_name):
    
    commentID = obj['id']
    authorID, authorname = obj["author_fullname"], obj['author']
    creation_ts = obj['created_utc']
    
    
    creation_date = None
    if 'date' not in obj:
        creation_date = convert_timestamp(creation_ts)
    else:
        creation_date = obj['date']
    
    
    score, awards_received = obj["total_awards_received"], obj["score"]
    
    info = {
        'key' : unique_id,
        'commentID' : commentID,
        'comment' : final_comment,
        'authorID' : authorID,
        'authorname' : authorname,
        'classname' : class_name,
        'classno' : class_no,
        'creationdate' : creation_date,
        'creationts' : creation_ts,
        'score' : score,
        'awards_received' : awards_received
    }
    
    info_arr = [unique_id, commentID, final_comment, authorID, authorname, class_name, class_no, 
                creation_date, creation_ts, score, awards_received]
    
    return info, info_arr

def create_dataset():
    full_info_dict = {}
    full_info_arr = []
    
    headings =   ['key', 'commentID', 'comment', 'authorID' , 'authorname' , 'classname', 'classno', 
                  'creationdate', 'creationts','score','awardsreceived','nftokens']
    
    full_info_arr.append(headings)
    
    for author_no, author in enumerate(authors):
        
        print('--------------------', author, author_no, '---------------------')
        
        # Creating the file path for this class
        filepath = input_data_folder + author
        if synthetic == True: filepath  += 'GPT2Bot'
        filepath += '.txt'
        objects = read_json_objects(filepath)
        
        
        class_no, classname = author_no,author
        
        full_info_dict[author_no] = {}
        full_info_dict[author_no]['classno'] = author_no
        full_info_dict[author_no]['classname'] = author
        full_info_dict[author_no]['data'] = {}
        
        
        for comment_no, obj in enumerate(objects):
            
            comment = obj['body']
            unique_ID = create_unique_ID(class_no, comment_no)
            processed_comment, nftokens = preprocess(comment)
            info_dict, info_arr = extract_data(processed_comment, obj, unique_ID, class_no, classname)
            
            info_dict['nftokens'] = nftokens
            info_arr.append(nftokens)
            
            full_info_dict[author_no]['data'][unique_ID] = info_dict
            
            full_info_arr.append(info_arr)
            
            if comment_no % 1000 == 0:
                
                print(json.dumps(info_dict, indent = 4))
                
                if DUMP_MODE == True:
                    dumpJsonFile(full_info_dict, dict_output_filepath, verbose = False, print_dict = False)
                    dumpPickleFile(full_info_arr, arr_output_filepath, verbose = False)

              
create_dataset()              
            
                      
                      
                    
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            


    

    
    