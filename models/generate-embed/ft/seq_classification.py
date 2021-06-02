import torch
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertModel, RobertaTokenizer, TFRobertaModel, RobertaModel
from transformers import XLNetTokenizer,TFXLNetModel,OpenAIGPTTokenizer,OpenAIGPTModel,GPT2Tokenizer,GPT2Model,TFOpenAIGPTModel,TFGPT2Model
from transformers import BertForSequenceClassification,XLNetForSequenceClassification,AdamW, BertConfig,RobertaForSequenceClassification,RobertaForSequenceClassification
from transformers import AutoTokenizer
import pickle
import wget
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch._utils import _accumulate
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
from pathlib import Path
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import classification_report
import argparse
import math
import matplotlib.pyplot as plt
import  sys
import os


from SeqClass import *
sys.path.insert(0,'../../')

from read_write_functions import *


def load_dataset(dataset_path, ids_path):
    """
    Loads the dataset
    """
    ids = loadJsonFile(ids_path)
    complete_dataset = loadJsonFile(dataset_path)
    
    train_df = make_dataset(ids['train'], complete_dataset)
    val_df = make_dataset(ids['val'], complete_dataset)
    test_df = make_dataset(ids['test'], complete_dataset)
    
    return train_df, val_df, test_df

def make_dataset(ids, complete_dataset):
    """
    Curates the dataset in a DataFrame Format
    """
    dataset = []
    
    for commentID  in ids:
        
        label1 = commentID.split('-')[0]; label1 = label1[1:]
        commentinfo = complete_dataset[label1]['data'][commentID]
        
        label2 = commentinfo['classno']
        comment = commentinfo['comment']
        
        assert(commentinfo['nftokens'] >= min_nf_tokens)
        assert(int(label1) == label2)
        
        dataset.append([comment, label2])
    
    print('# Nfsamples = ', len(dataset) - 1)
    
    assert(len(dataset) == len(ids))
    
    df = pd.DataFrame(data = dataset, columns=['sentence', 'label'])
    
    return df
    



def create_folders(folder_path):
    """
    Hierarchically creates folders
    """
    path = Path(folder_path)
    path.mkdir(parents = True,exist_ok = True)

# Assigns GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def check_gpu():
    # Checking GPU Usage. Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    print(device_name)

    #The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        pass
        #raise SystemError('GPU device not found')

    device = None

    if torch.cuda.is_available():       
        device = torch.device("cuda");print('There are %d GPU(s) available.' % torch.cuda.device_count());print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu");print('No GPU available, using the CPU instead.')
    return device

device = check_gpu()

#-----------------PARAMS----------------------------------

#TOKENS taken for a sentence
maximum_length = 75 

# The hidden size of the GPT2 and GPT models
hidden_size = maximum_length * 768

# Lowercase the sentence for embedding generation
lowercase_sentence = False

# Model Type
model_name = 'roberta-base'
model_type = 'cnn'

# Dataset Configurations
nf_train_samples = 600
nf_val_samples = 100
nf_test_samples = 200
num_classes = 108

# May vary based on size of GPU
min_nf_tokens = 6



DUMP_MODE = True
loss_type = 'cross_entropy'

# If the labels are one-hot encoded are not
one_hot = False
    
#------------------------------------------PATHS----------------------------------------------------------
path_str = str(num_classes) + '_' + str(nf_train_samples) + '_' + str(nf_val_samples) + '_' + str(nf_test_samples)
working_folder = "../../../"

#---------Input Folder-----------
input_ids_path = working_folder + 'dataset/synthetic/splits/' + str(min_nf_tokens) + '/' + path_str + '_dataset.json'
complete_dataset_path = working_folder + 'dataset/synthetic/' + 'dataset.json'

train_df, val_df, test_df = load_dataset(complete_dataset_path, input_ids_path)

print(train_df)
print(val_df)
print(test_df)

#---------Output folder-----------
store_folder = working_folder + 'results/generate-embed/ft/' + model_name + '/' + model_type + '/'
store_folder += str(min_nf_tokens) + '/'

models_folder = store_folder + 'models/' + model_name + '/' + loss_type + '/'
model_weights_folder = store_folder + 'model_weights/' + model_name + '/' + loss_type + '/'

# epoch_results_folder = store_folder + 'per_epoch_results/' + model_name + '/' + loss_type + '/' + path_str + '/'
training_stats_folder = store_folder + 'per_epoch_stats/' + model_name + '/' +  loss_type + '/' 
training_stats_path = training_stats_folder + path_str + '.json'
best_results_folder = store_folder + 'results/best/' + model_name + '/' +  loss_type + '/'+ path_str + '/'

finetuned_embeddings_folder = store_folder + 'finetuned_embedding/best/' + model_name + '/' + loss_type + '/' + path_str + '/'

best_predictions_folder = store_folder + 'predictions/best/' + model_name + '/' + loss_type + '/'
best_predictions_folder +=  path_str + '/'

graph1_folder = store_folder + 'avg_graph/best/' + model_name + '/' + loss_type + '/'
graph2_folder = store_folder + 'class_wise_avg_graph/best/' + model_name + '/' + loss_type + '/'

graph1_path = graph1_folder + path_str + '.png'
graph2_path = graph2_folder + path_str + '.png'

final_model_path = models_folder + 'final/' + path_str + '.pkl'
best_model_path = models_folder + 'best/' + path_str + '.pkl'

final_model_weights_path = model_weights_folder + 'final/' + path_str + '.pkl'
best_model_weights_path = model_weights_folder + 'best/' + path_str + '.pkl'

#----------------------------------------------------
print("Model Name = ",model_name)
print("Number of classes = ",num_classes)
print("Number of train samples per class =",nf_train_samples)
print("Number of val samples per class =",nf_val_samples)
print("Number of test samples per class =",nf_test_samples)
#------------------------------------------------------

if DUMP_MODE == True:
#     create_folders(epoch_results_folder)
    create_folders(best_results_folder)
    create_folders(best_predictions_folder)
    create_folders(graph1_folder)
    create_folders(graph2_folder)
    create_folders(models_folder + 'final/')
    create_folders(models_folder + 'best/')
    create_folders(model_weights_folder + 'final/')
    create_folders(model_weights_folder + 'best/')
    create_folders(finetuned_embeddings_folder + 'best/')
    create_folders(training_stats_folder)
#-----------------------------------------------------


def get_dist(labels, nfclass):
	per_class_count = [0 for i in  range(nfclass)]
	for label in labels : per_class_count[label] += 1
	print('--- Class Distribution ---')
	print(per_class_count)
	return per_class_count

def get_one_hot_labels(labels, num_classes = -1):
	one_hot_labels = torch.nn.functional.one_hot(labels, num_classes)
	#print('The shape of One Hot Labels Array = ',one_hot_labels.shape)
	return one_hot_labels


def get_dataset(balanced = True, num_classes = 2, nf_train =  1600,nf_test = 400):
    """
    Fetches the dataset 
    """
    filename =  'dataset_for_finetune_' + str(num_classes) + '_' + str(nf_train) + '_' + str(nf_test) + '_complete_comments.csv' 
    datasets_folder = working_folder + 'pipeline/experiment_3/supervised_closed_world_1/datasets_for_finetuned/store/'
    print(filename)
    dataframe_name = datasets_folder + '/dataset_for_finetune_30_800_200_complete_comments.csv'
    if os.path.exists(datasets_folder + filename) == True:
        print(datasets_folder  + filename, 'is correct')
    
    df = pd.read_csv(datasets_folder  + filename, index_col = 0)
    return df


print('Loading the dataset......')

# The dataset should necessarily have 'sentence' and 'label' column 
# df = get_dataset(balanced = balanced, num_classes = num_classes, nf_train = nf_train_samples ,nf_test = nf_test_samples)
# print(df.head())
# print('Shape of Dataframe = ',df.shape)

# import sys
# sys.exit(0)

def get_ind_dataset(df):
    
    sentences = df.sentence.values
    labels = list(df.label.values)
    per_class_count = get_dist(labels, num_classes)
    
    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    
    if one_hot == True:
        one_hot_labels = get_one_hot_labels(labels, num_classes)
        labels = one_hot_labels
        
    return sentences, labels

train_sentences, train_labels = get_ind_dataset(train_df)
val_sentences, val_labels = get_ind_dataset(val_df)
test_sentences, test_labels = get_ind_dataset(test_df)

print('Loading Tokenizer .....')
tokenizer = load_tokenizer(model_name)
print('Loading Model .....')
model = None


if model_type == "cnn":
    model = load_model_cnn(model_name, num_classes, hidden_size)
else:
    model = load_model(model_name, num_classes, hidden_size)


print(type(model))
print(model)


def add_special_token(model, tokenizer,special_token_key = 'pad_token',special_token = '[PAD]'):
    tokenizer.add_special_tokens({special_token_key: special_token})
    model.ptmodel.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

print('If model name is gpt2 or gpt then, special Padding tokens need to be added and the model needs to be made aware of that')
if model_name == 'gpt2' or model_name == 'openai-gpt':
    model, tokenizer = add_special_token(model, tokenizer, special_token_key = 'pad_token',special_token = '[PAD]')

    
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __add__(self, other):
        return ConcatDataset([self, other])

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)

    
def get_additional_info(sentences, labels, lowercase = False):
    input_ids = []
    attention_masks = []
    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    
    for sent in sentences:
        
        sent = str(sent)
        
        if lowercase == True : 
            sent = sent.lower()

        encoded_dict = tokenizer.encode_plus(sent,truncation = True, add_special_tokens = True, 
                                             max_length = maximum_length, pad_to_max_length = True, 
                                             return_attention_mask = True,return_tensors = 'pt')

        input_ids.append(encoded_dict['input_ids'])
        
        attention_masks.append(encoded_dict['attention_mask'])
        
        #max_len = max(max_len, len(encoded_dict['input_ids']))
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)
    
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    print('Attention mask:',attention_masks[0])
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset

print('Tokenize all of the sentences and map the tokens to their word IDs .....')


train_dataset = get_additional_info(train_sentences, train_labels, lowercase = False)
test_dataset = get_additional_info(test_sentences, test_labels, lowercase = False)
val_dataset = get_additional_info(val_sentences, val_labels, lowercase = False)


def per_class_distribution(dataset, nfclasses = 30,one_hot = False):
    per_class_count = [0 for i in range(nfclasses)]
    Y = [];X = []
    c = 0 
    for sample in dataset:
        y = sample[2].numpy()
        if one_hot == True:
            y = int(np.argmax(np.array(y).reshape(-1,1), axis = 0))
        Y.append(int(y));X.append(1)
        per_class_count[y] += 1
        c += 1
    
    print('------Per class count------')
    return per_class_count,X,Y

print('{:>5,} Train samples'.format(len(train_dataset)))
print('{:>5,} Test samples'.format(len(test_dataset)))
print('{:>5,} Validation samples'.format(len(val_dataset)))
   

# The DataLoader needs to know our batch size for training, so we specify it here. 
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
batch_size = 48

print('Creating the train dataloader with a batch size ',batch_size,'.....')

# Create the DataLoaders for our training and validation sets. 
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size, shuffle = False)

# Sequential Sampler is used instead of Random Sampler
# The batch size of test dataloader is nf_test_samples * num_classes to process the whole thing as 1 batch
print('Creating the test dataloader with a batch size',nf_test_samples * num_classes ,'.....')
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = 1 , shuffle = False)

print('Creating the val dataloader with a batch size.....')
val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = 1 , shuffle = False)

print("Summary of the model's parameters as a list of tuples.")
params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:-4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
				lr = 5e-5, # default is 5e-5,
				eps = 1e-8 # default is 1e-8.
                )

# Number of training epochs. The BERT authors recommend between 2 and 4. I ran for maximum 7
epochs = 7
# Total number of training steps is [number of batches] x [number of epochs]. (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):
    """
    Calculates the accuracy
    """
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    results = classification_report(labels_flat, pred_flat, digits = 5,output_dict = False)
    results_json = classification_report(labels_flat, pred_flat, digits = 5,output_dict = True)
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
    return accuracy, results, results_json, pred_flat 

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


model = model.cuda()
device = check_gpu()

# This training code is based on the `run_glue.py` script here: https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

def seed_everything(seed):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

seed_everything(seed_val) 

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

model_after_2_epochs = None
best_model = None
best_accuracy = 0
best_loss = 9999999
criterion = None

if loss_type == 'multilabel':
    criterion = torch.nn.MultiLabelSoftMarginLoss()

def organize_embeddings(hidden_states, model_name):
    final_embeddings = []
    c = 0
    for batch_embeddings in hidden_states:
        batch_embed_numpy = batch_embeddings
        for embed in batch_embed_numpy: 
            if model_name[:4] == 'bert' or model_name[:4] == 'robe':
                # Classification Token is the first token
                final_embeddings.append(embed)
            elif model_name[:4] == 'xlne':
                # Classification Token is the last token
                final_embeddings.append(embed[-1])
            else:
                # Pooled and given the output (Self Made)
                final_embeddings.append(embed)
            if c % 1000 == 0:
                print(c,batch_embed_numpy.shape,embed.shape)
            c += 1
    final_embeddings = np.array(final_embeddings).reshape(-1,1)
    return final_embeddings
    
def evaluate(model, dataloader, model_name):
    """
    Evaluating the model
    """
    
    short_name = model_name[:4]
    
    print('Evaluating the model ......')
    
    # Start time
    t0 = time.time()
    
    # Put the model in evaluation mode-- the dropout layers behave differently during evaluation.
    model.eval()
    
    # Tracking variables 
    total_eval_accuracy = 0;total_eval_loss = 0;nb_eval_steps = 0
    
    # Initializing all storage Units
    all_logits = [];all_label_ids = [];all_hidden_states = []
    
    # Batch
    batch_no = 0
    
    for batch in dataloader:
        
        # Loading into Device (GPU)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2]
        
        with torch.no_grad():
            if loss_type == 'multilabel': 
                b_labels = np.argmax(b_labels, axis = 1)
                
            b_labels = b_labels.to(device)
            # Loading into model ang getting the results
            loss, logits, hidden = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
            
            # For multilabel loss
            if loss_type == 'multilabel':
                b_labels_new = b_labels.cpu()
                b_labels_new = get_one_hot_labels(b_labels_new, num_classes)
                b_labels_new = b_labels_new.to(device)
                loss = criterion(logits, b_labels_new)
        
        # Evaluation Loss 
        total_eval_loss += loss.item()
        
        # Logits 
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Output for Last hidden layer
        if short_name == 'bert' or short_name == 'robe' or short_name == 'xlne':
            all_hidden_states.append(hidden[-1].to('cpu').numpy())
        
        # With GPT2 we pretty much made our own custom layer
        elif short_name == 'open' or short_name == 'gpt2':
            all_hidden_states.append(hidden.to('cpu').numpy())
        
#         if batch_no % 1000 == 0 : 
#             print(label_ids.shape, logits.shape, len(all_hidden_states),all_hidden_states[0].shape)
            #print(label_ids.shape, logits.shape)
        
        if batch_no == 0 :
            all_label_ids, all_logits = label_ids, logits#, hidden_states
        else:
            all_label_ids = np.concatenate((all_label_ids,label_ids), axis = 0)
            all_logits = np.concatenate((all_logits, logits), axis = 0)
            #all_hidden_states = np.concatenate((all_hidden_states,logits), axis = 0)
        
        if batch_no % 1000 == 0 :
            print(batch_no, all_label_ids.shape, all_logits.shape, len(all_hidden_states),all_hidden_states[0].shape)
            #print(batch_no, all_label_ids.shape, all_logits.shape)
        
        batch_no += 1 
    
#     all_hidden_states = all_hidden_states.detach().cpu().numpy()
    final_embeddings = organize_embeddings(all_hidden_states, model_name)
#     print(label_ids.shape, logits.shape, final_embeddings.shape)
    
    # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
    eval_accuracy, results, results_json, predictions = flat_accuracy(all_logits, all_label_ids)
    accuracy = results_json['accuracy']
    print(results)
    
    # Calculate the average loss over all of the batches.
    loss = total_eval_loss / len(dataloader)
    
    # Measure how long the validation run took.
    end_time = time.time() 
    test_time = format_time(end_time - t0)
    
    print("\t Accuracy: {0:.5f}".format(accuracy))
    print("\t Loss: {0:.5f}".format(loss))
    print("\t Timing took: {:}".format(test_time))
    print("\t Confirming Accuracy:", eval_accuracy)
    print("\t Number of Samples :")
    print(label_ids.shape, logits.shape, final_embeddings.shape)
    
    return results_json, predictions, final_embeddings, all_label_ids, all_logits, loss, test_time 
    

# For each epoch...
for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to `train` just changes the *mode*, 
    # it doesn't *perform* the training. `dropout` and `batchnorm` layers behave differently during training vs. test 
    #(source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 100 == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        # Unpack this training batch from our dataloader. 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method. `batch` contains three pytorch tensors: [0]: input ids, [1]: attention masks, [2]: labels 

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2]

        # print(b_input_ids.shape,b_labels.shape)
        # Always clear any previously calculated gradients before performing a backward pass. 
        # PyTorch doesn't do this automatically because accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)

        model.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments arge given and what flags are set. 
        # For our useage here, it return the loss (because we provided labels) 
        # and the "logits"--the model outputs prior to activation.
        
        if loss_type == 'multilabel' : b_labels = np.argmax(b_labels, axis = 1)
        b_labels = b_labels.to(device)
        
        loss, logits, hidden = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
        
        # For multilabel loss
        if loss_type == 'multilabel':
            b_labels = b_labels.cpu()
            b_labels = get_one_hot_labels(b_labels, num_classes)
            b_labels = b_labels.to(device)
            loss = criterion(logits, b_labels)
        
        # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end. 
        # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
        total_train_loss += loss.item()
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient. 
        # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, 
        # the learning rate, etc.
        optimizer.step()
        scheduler.step() # Update the learning rate.

    # Calculate the average loss over all of the batches.
    train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    
    print("  Average training loss: {0:.2f}".format(train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Test
    # ========================================
    # After the completion of each training epoch, measure our performance our Test set.
    
    print('Evaluating train dataset ....')
    train_results_json, train_predictions, train_final_embeddings, train_label_ids, train_logits, e_train_loss, r_time = evaluate(model, train_dataloader,model_name)
    
    print('Evaluating validation dataset ....')
    val_results_json, val_predictions, val_final_embeddings, val_label_ids, val_logits, val_loss, val_time = evaluate(model, val_dataloader,model_name)
    
    print('Evaluating test dataset ....')
    test_results_json, test_predictions, test_final_embeddings, test_label_ids, test_logits, test_loss, test_time = evaluate(model, test_dataloader,model_name)
    
    train_accuracy = train_results_json['accuracy']
    test_accuracy = test_results_json['accuracy']
    val_accuracy = val_results_json['accuracy']
    
    # Record all statistics from this epoch.
    if test_accuracy > best_accuracy:
        best_model = model
        best_accuracy = test_accuracy
        best_loss = test_loss
        
        if DUMP_MODE == True:
            dumpJsonFile(test_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_test.json')
            dumpJsonFile(val_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_val.json')
            dumpJsonFile(train_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_train.json')
            dumpPickleFile([test_label_ids, test_final_embeddings,
                            test_logits],best_predictions_folder +'epoch_'+ str(epoch_i)+'_predictions_test.pkl')
            dumpPickleFile([val_label_ids, val_final_embeddings,
                            val_logits],best_predictions_folder +'epoch_'+ str(epoch_i)+'_predictions_val.pkl')
            dumpPickleFile([train_label_ids,train_final_embeddings,train_logits],
                           best_predictions_folder+'epoch_'+str(epoch_i)+'_predictions_train.pkl')
            
            make_precision_recall_curve(test_logits, test_label_ids, num_classes, savefigpath1 = graph1_path, 
                                        savefigpath2 = graph2_path, probs = False)
            
            torch.save(model.state_dict(), best_model_weights_path)
            torch.save(model, best_model_path)

    elif test_accuracy == best_accuracy:
        if test_loss < best_loss:
            best_model = model;best_accuracy = avg_test_accuracy;best_loss = avg_test_loss
            
            if DUMP_MODE == True:
                dumpJsonFile(test_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_test.json')
                dumpJsonFile(val_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_val.json')
                dumpJsonFile(train_results_json, best_results_folder + 'epoch_' + str(epoch_i) + '_results_train.json')
                
                dumpPickleFile([test_label_ids, test_final_embeddings,
                                test_logits],best_predictions_folder+'epoch_'+str(epoch_i)+'_predictions_test.pkl')
                
                dumpPickleFile([train_label_ids,train_final_embeddings,train_logits],
                               best_predictions_folder+'epoch_'+str(epoch_i)+'_predictions_train.pkl')
                
                dumpPickleFile([val_label_ids, val_final_embeddings,
                            val_logits],best_predictions_folder +'epoch_'+ str(epoch_i)+'_predictions_val.pkl')
                
                make_precision_recall_curve(test_logits, test_label_ids, num_classes, savefigpath1 = graph1_path, 
                                            savefigpath2 = graph2_path, probs = False)
                
                torch.save(model.state_dict(), best_model_weights_path)
                torch.save(model, best_model_path)

    torch.save(model.state_dict(), final_model_weights_path)
    torch.save(model, final_model_path)

                
    training_stats = {
            'epoch': epoch_i + 1,
            'Training Loss': train_loss,
            'Test Loss': test_loss,
            'Test Accurary': test_accuracy,
            'Train Accurary': train_accuracy,
            'Training Time': training_time,
            'Testing Time': test_time,
            'Val Loss' : val_loss,
            'Val Accuracy' : val_accuracy,
            'Val Time' : val_time
        }

    print('Dumping training folders .....')
    
    if DUMP_MODE == True:
        pass
#         dumpJsonFile(training_stats, training_stats_path)

print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

for batch in test_dataloader:
    print(batch[2].numpy().shape)

    

    

# from subprocess import call
# call(["python3", "seq_classification_bert.py"])
# def get_hidden_states(model, dataloader, model_name):
#     hidden_states = []
#     short_name = model_name[:4]
#     model.eval()
#     for batch in dataloader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2]
#         with torch.no_grad():
#             if loss_type == 'multilabel':
#                 b_labels = np.argmax(b_labels, axis = 1)
#             b_labels = b_labels.to(device)
#             (loss, logits,hidden) = model(b_input_ids, attention_mask=b_input_mask, labels = b_labels)
#         if short_name == 'bert' or short_name == 'robe' or short_name == 'xlne':
#             # Output for Last hidden layer
#             hidden_states.append(hidden[-1])
#         elif short_name == 'open' or short_name == 'gpt2':
#             hidden_states.append(hidden)
#     return hidden_states

# test_hidden_states = get_hidden_states(best_model, test_dataloader,model_name)



# test_embeddings = organize_embeddings(test_hidden_states, model_name)
# dumpPickleFile(test_embeddings, best_model_test_embeddings_path)

# dumpPickleFile(model, best_model_path)
# torch.save(model.state_dict(), best_model_weights_path)


# import pandas as pd
# def display_stats(training_stats):
#     # Display floats with two decimal places.
#     pd.set_option('precision', 2)
#     # Create a DataFrame from our training statistics.
#     df_stats = pd.DataFrame(data=training_stats)
#     # Use the 'epoch' as the row index.
#     df_stats = df_stats.set_index('epoch')
#     # Display the table.
#     return df_stats
# df_stats.to_csv()

# def plot_everything():
#     # Use plot styling from seaborn.
#     sns.set(style='darkgrid')
#     # Increase the plot size and font size.
#     sns.set(font_scale=1.5)
#     plt.rcParams["figure.figsize"] = (12,6)
#     # Plot the learning curve.
#     plt.plot(df_stats['Training Loss'], 'b-o', label = "Training Loss")
#     plt.plot(df_stats['Test Accur.'], 'r-o', label = " Test Accuracy")
#     plt.plot(df_stats['Test Loss'], 'g-o', label = " Test Loss")
#     # Label the plot.
#     plt.title("Training & Test Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.xticks([1, 2, 3, 4])
#     plt.show()