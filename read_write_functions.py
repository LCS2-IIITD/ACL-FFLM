import json
import pickle
import csv
import sys
import jsonlines
import datetime
import numpy as np
import random
import os
import torch
import pathlib

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_folders_path(path_name):
    path = pathlib.Path(path_name)
    path.parent.mkdir(parents = True,exist_ok = True)
    

def loadPickleFile(filepath):
	#print("Loading the pickle file from",filepath,"...........")
	pickle_in = open(filepath,"rb")
	example_dict = pickle.load(pickle_in)
	#print("Loaded the pickle File")
	return example_dict

def openCSVfile(filepath, delimiter = ","):
    """
    Returns the lists for csv file 
    """
    with open(filepath,"r") as csvfile:
        rows =  csv.reader(csvfile,delimiter = delimiter)
        return list(rows)

def dumpPickleFile(data,filepath):
	pickle_out = open(filepath,"wb")
	#print("Dumping the Pickle file into ",filepath,"...........")
	pickle.dump(data, pickle_out)
	#print("Dumped the pickle File")
	pickle_out.close() 

def dumpJsonFile(dictionary,filepath):
	#print("Dumping a dictionary to filepath",filepath,"...............")
	with open(filepath,"w+") as jsonFile:
		json.dump(dictionary,jsonFile,indent=4,sort_keys =True)
	#print("Dumped Successfully",filepath)

def loadJsonFile(filepath):
	#print("Loading a dictionary to filepath",filepath,"...............")
	dictionary = {}
	with open(filepath) as jsonFile:
		dictionary = json.load(jsonFile)
	#print("Loaded Successfully")
	return dictionary

# Convert timestamp
def convert_timestamp(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    date_split = date.split("-");month = int(date_split[1]);year = int(date_split[0])
    return month,year,date

def read_json_objects(filepath):
    objects = [] 
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            created_timestamp = obj['created_utc']#;updated_timestamp = obj['updated_utc']
            month,year,date = convert_timestamp(created_timestamp)
            objects.append([obj,month,year,date])
    return objects


def get_arguments():
    n = len(sys.argv) 
    if n != 4:
        print('Incomplete not executing')
        sys.exit(0)
    print("Total arguments passed:", n) 
    print("\nName of Python script:", sys.argv[0]) 
    print("\nArguments passed:", end = " ") 
    
    model_name = sys.argv[1]
    class_name = sys.argv[2]
    json_filename = sys.argv[3]
    save_filename = sys.argv[4]
    
    return model_name, json_filename, save_filename

