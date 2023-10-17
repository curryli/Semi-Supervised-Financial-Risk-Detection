# coding: UTF-8
import torch
from torch.utils import data
from huggingface.tokenization import BertTokenizer
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import random

# def tokenize_sentence(tokenizer=None,sentence =None,std_size =None,PAD =0):
#     '''
#     '''

#     token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
#     seq_len = len(token_ids)
#     if std_size:
#         if len(token_ids) < std_size:
#             token_ids.extend([PAD] * (std_size - seq_len))
#         else:
#             token_ids = token_ids[:std_size]
#             seq_len = std_size
#     return token_ids


def load_sigir_pandas_multi_select(filepath, std_size=2000, sep='\t'):
    nodes_feat= pd.read_csv(filepath, header=0, sep='\t', error_bad_lines=False)   #平衡后的

    X =  [[int(x) for x in xstr.split(",")] for xstr in nodes_feat['X'].values.tolist()]

    M = nodes_feat['M'].values.tolist()

    Y =  [[int(y) for y in xstr.split(",")] for xstr in nodes_feat['Y'].values.tolist()]

    return X,Y,M



class Dataset(torch.utils.data.Dataset):
  
  def __init__(self, X, Y):
        'Initialization'
        self.Y = Y
        self.X = X

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.X[index]
        x = torch.LongTensor(x)
        y = torch.LongTensor(self.Y[index])

        return x, y
    

class DatasetGT(torch.utils.data.Dataset):
  
  def __init__(self, X, Y, G):
        'Initialization'
        self.X = X
        self.Y = Y
        self.G = G

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.LongTensor(self.X[index])
        y = torch.LongTensor(self.Y[index])
        g = torch.FloatTensor(self.G[index])

        return x, y, g

    
if __name__ == "__main__":
    pass