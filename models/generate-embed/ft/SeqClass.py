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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
import matplotlib.pyplot as plt

class GPT2SequenceClassifier(torch.nn.Module):
    def __init__(self, hidden_size: int,num_classes:int ,max_seq_len:int,gpt_model_name:str):
        super(GPT2SequenceClassifier,self).__init__()
        self.ptmodel = GPT2Model.from_pretrained(gpt_model_name)
        self.fc0 = torch.nn.Linear(hidden_size, 768)
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()
        
    def forward(self, x_in, attention_mask,labels = None):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        hidden_vector = self.fc0(gpt_out.view(batch_size,-1)) #(batch_size , max_len, num_classes)
        prediction_vector = self.fc1(hidden_vector)
        #prediction_vector = self.softmax(prediction_vector)
        if labels != None:
            loss = self.loss_func(prediction_vector,labels)
            return loss,prediction_vector,hidden_vector
        else:
            return prediction_vector,hidden_vector

class GPTSequenceClassifier(torch.nn.Module):
    def __init__(self, hidden_size: int,num_classes:int ,max_seq_len:int,gpt_model_name:str):
        super(GPTSequenceClassifier,self).__init__()
        self.ptmodel = OpenAIGPTModel.from_pretrained(gpt_model_name)
        self.fc0 = torch.nn.Linear(hidden_size, 768)
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_in, attention_mask, labels = None):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        hidden_vector = self.fc0(gpt_out.view(batch_size,-1)) #(batch_size , max_len, num_classes)
        prediction_vector = self.fc1(hidden_vector)
        #prediction_vector = self.softmax(prediction_vector)
        if labels != None:
            loss = self.loss_func(prediction_vector,labels)
            return loss,prediction_vector,hidden_vector
        else:
            return prediction_vector,hidden_vector


class SequenceClassifier(torch.nn.Module):
    def __init__(self, hidden_size: int,num_classes:int ,max_seq_len:int,model_name:str):
        super(SequenceClassifier,self).__init__()
        model = None
        short_name = model_name[:4]
        
        if short_name == "bert":
            model = BertModel.from_pretrained(model_name,attention_mask = attention_mask)
        elif short_name == "robe":
            model = RobertaModel.from_pretrained(model_name,attention_mask = attention_mask)
        elif short_name == "xlne":
            model = XLNetModel.from_pretrained(model_name, num_labels = nflabels)
        elif short_name == 'open':
            model = OpenAIGPTModel.from_pretrained(model_name)
        elif short_name == 'gpt2':
            self.ptmodel = GPT2Model.from_pretrained(model_name)

        self.ptmodel = model
        self.fc0 = torch.nn.Linear(hidden_size, 768)
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_in, attention_mask, labels):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        hidden_vector = self.fc0(gpt_out.view(batch_size,-1)) #(batch_size , max_len, num_classes)
        prediction_vector = self.fc1(hidden_vector)
        #prediction_vector = self.softmax(prediction_vector)
        
        loss = self.loss_func(prediction_vector,labels)
        return loss,prediction_vector,hidden_vector

class SequenceClassifierCNN(torch.nn.Module):
    def __init__(self, hidden_size: int,num_classes:int ,max_seq_len:int,model_name:str):
        super(SequenceClassifierCNN,self).__init__()
        model = None
        short_name = model_name[:4]
        
        if short_name == "bert":
            model = BertModel.from_pretrained(model_name,attention_mask = attention_mask)
        elif short_name == "robe":
            model = RobertaModel.from_pretrained(model_name)
        elif short_name == "xlne":
            model = XLNetModel.from_pretrained(model_name, num_labels = nflabels)
        elif short_name == 'open':
            model = OpenAIGPTModel.from_pretrained(model_name)
        elif short_name == 'gpt2':
            model = GPT2Model.from_pretrained(model_name)
        
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv1d(1,16, kernel_size = 2,stride = 1),
#             torch.nn.BatchNorm2d(16, track_running_stats = True),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.MaxPool1d(2, stride = 2, padding = 0,return_indices = False, ceil_mode = False))
        
        self.layer1 = torch.nn.Conv1d(75,16, kernel_size = 2,stride = 1)
        self.layer2 = torch.nn.BatchNorm1d(16, track_running_stats = True)
        self.layer3 = torch.nn.ReLU(inplace = True)
        self.layer4 = torch.nn.MaxPool1d(2, stride = 2, padding = 0,return_indices = False, ceil_mode = False)
        
        self.ptmodel = model
        hidden_size = 6128
        self.fc0 = torch.nn.Linear(hidden_size, 768)
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_in, attention_mask, labels):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        
        #print("gpt_out",gpt_out.shape)
#         prop_out = gpt_out.view(batch_size,-1)
#         print("prop_outA", prop_out.shape)
#         prop_out.unsqueeze_(-1)
#         prop_out = prop_out.expand(48,75,768)
#         print("prop_outB", prop_out.shape)
        prop_out1 = self.layer1(gpt_out)
        #print("prop_out1", prop_out1.shape)
        prop_out1 = self.layer2(prop_out1)
        #print("prop_out2", prop_out1.shape)
        prop_out1 = self.layer3(prop_out1)
        #print("prop_out3", prop_out1.shape)
        prop_out1 = self.layer4(prop_out1)
        #print("prop_out4", prop_out1.shape)
        
        prop_out = prop_out1.view(batch_size,-1)
        hidden_vector = self.fc0(prop_out) #(batch_size , max_len, num_classes)
        prediction_vector = self.fc1(hidden_vector)
        #prediction_vector = self.softmax(prediction_vector)
        loss = self.loss_func(prediction_vector,labels)
        return loss,prediction_vector,hidden_vector  

class SequenceClassifierCNNFreezed(torch.nn.Module):
    def __init__(self, hidden_size: int,num_classes:int ,max_seq_len:int,model_name:str):
        super(SequenceClassifierCNNFreezed,self).__init__()
        model = None
        short_name = model_name[:4]
        
        if short_name == "bert":
            model = BertModel.from_pretrained(model_name,attention_mask = attention_mask)
        elif short_name == "robe":
            model = RobertaModel.from_pretrained(model_name)
        elif short_name == "xlne":
            model = XLNetModel.from_pretrained(model_name, num_labels = nflabels)
        elif short_name == 'open':
            model = OpenAIGPTModel.from_pretrained(model_name)
        elif short_name == 'gpt2':
            model = GPT2Model.from_pretrained(model_name)
        
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv1d(1,16, kernel_size = 2,stride = 1),
#             torch.nn.BatchNorm2d(16, track_running_stats = True),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.MaxPool1d(2, stride = 2, padding = 0,return_indices = False, ceil_mode = False))
        
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
        
        self.layer1 = torch.nn.Conv1d(75,16, kernel_size = 2,stride = 1)
        self.layer2 = torch.nn.BatchNorm1d(16, track_running_stats = True)
        self.layer3 = torch.nn.ReLU(inplace = True)
        self.layer4 = torch.nn.MaxPool1d(2, stride = 2, padding = 0,return_indices = False, ceil_mode = False)
        
        self.ptmodel = model
        hidden_size = 6128
        self.fc0 = torch.nn.Linear(hidden_size, 768)
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_in, attention_mask, labels):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        
        #print("gpt_out",gpt_out.shape)
#         prop_out = gpt_out.view(batch_size,-1)
#         print("prop_outA", prop_out.shape)
#         prop_out.unsqueeze_(-1)
#         prop_out = prop_out.expand(48,75,768)
#         print("prop_outB", prop_out.shape)
        prop_out1 = self.layer1(gpt_out)
        #print("prop_out1", prop_out1.shape)
        prop_out1 = self.layer2(prop_out1)
        #print("prop_out2", prop_out1.shape)
        prop_out1 = self.layer3(prop_out1)
        #print("prop_out3", prop_out1.shape)
        prop_out1 = self.layer4(prop_out1)
        #print("prop_out4", prop_out1.shape)
        
        prop_out = prop_out1.view(batch_size,-1)
        hidden_vector = self.fc0(prop_out) #(batch_size , max_len, num_classes)
        prediction_vector = self.fc1(hidden_vector)
        #prediction_vector = self.softmax(prediction_vector)
        
        
#         print(labels)
#         print(labels.shape)
#         print(prediction_vector)
#         print(prediction_vector.shape)
        
        loss = self.loss_func(prediction_vector, labels)
        
        return loss, prediction_vector, hidden_vector
        
    

def load_model_cnn_freeze(model_name, nflabels, hidden_size = None):
    print('Loading  Model...')
    short_name = model_name[:4]
    model = None
    model1 = None
    if short_name == "bert":
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels = nflabels,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "robe":
        model = SequenceClassifierCNNFreezed(hidden_size = hidden_size, num_classes = nflabels,model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "xlne":
        model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels = nflabels ,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'open':
        model = GPTSequenceClassifier(hidden_size = hidden_size, num_classes = nflabels,gpt_model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'gpt2':
        model = SequenceClassifierCNNFreezed(hidden_size = hidden_size, num_classes = nflabels,model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    else:
        raise Exception("Model Name is incorrect")
        
    
    return model


def load_model_cnn(model_name, nflabels, hidden_size = None):
    print('Loading  Model...')
    short_name = model_name[:4]
    model = None
    model1 = None
    if short_name == "bert":
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels = nflabels,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "robe":
        model = SequenceClassifierCNN(hidden_size = hidden_size, num_classes = nflabels,model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "xlne":
        model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels = nflabels ,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'open':
        model = GPTSequenceClassifier(hidden_size = hidden_size, num_classes = nflabels,gpt_model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'gpt2':
        model = SequenceClassifierCNN(hidden_size = hidden_size, num_classes = nflabels,model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    else:
        raise Exception("Model Name is incorrect")
        
    
    return model


def load_model(model_name, nflabels, hidden_size = None):
    print('Loading  Model...')
    short_name = model_name[:4]
    model = None
    model1 = None
    if short_name == "bert":
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels = nflabels,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "robe":
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = nflabels , output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == "xlne":
        model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels = nflabels ,output_attentions = False,output_hidden_states = True)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'open':
        model = GPTSequenceClassifier(hidden_size = hidden_size, num_classes = nflabels,gpt_model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    elif short_name == 'gpt2':
        model = GPT2SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels,gpt_model_name = model_name, max_seq_len = 64)
        #model1 = SequenceClassifier(hidden_size = hidden_size, num_classes = nflabels, model_name = model_name, max_seq_len = 64)
    else:
        raise Exception("Model Name is incorrect")
    return model


def load_tokenizer(model_name = "bert-base-uncased"):
    print('Loading  tokenizer...')
    tokenizer = None
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(model_name,progress=False)
    elif model_name == "bert-base-cased":
        tokenizer = BertTokenizer.from_pretrained(model_name,progress=False)
    elif model_name == 'roberta-base':
        tokenizer =  RobertaTokenizer.from_pretrained(model_name,progress=False)
    elif model_name == 'xlnet-base-cased':
        tokenizer = XLNetTokenizer.from_pretrained(model_name,progress=False)
    elif model_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt',progress = False)
    elif model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2',do_lower_case=True,progress=False)
    return tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer



def make_precision_recall_curve(test_probs, test_labels, num_classes, savefigpath1 = "", savefigpath2 = "", probs = True ):
    
    classes = list(range(num_classes))
    
    print(classes)
    
    
    Y_test = []
    
    if probs == False:
        # Applying soft max if it is not already (due to logits from NN)
        test_probs = np.exp(test_probs)/sum(np.exp(test_probs))
    
    
    if num_classes == 2:
        for i in test_labels:
            if i == 0:
                Y_test.append([1,0])
            else:
                Y_test.append([0,1])
        Y_test = np.array(Y_test)
    else:
        Y_test = label_binarize(test_labels, classes = classes)

    n_classes = num_classes

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    print(np.array(test_probs).shape)
    print(np.array(Y_test).shape)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],test_probs[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], test_probs[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), test_probs.ravel())
    average_precision["micro"] = average_precision_score(Y_test, test_probs, average = "micro")
    
    # 
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.savefig(savefigpath1)
    plt.show()
    
    colors = cycle(['green','red','blue','darkorange', 'olive','cornflowerblue','navy', 'turquoise'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color = color, lw = 2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom = 0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve, multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.savefig(savefigpath2)
    plt.show()