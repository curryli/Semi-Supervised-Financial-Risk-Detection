# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from dgl.data.utils import load_graphs
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import random
import numpy as np
np.random.seed(0)

import pandas as pd

import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch import edge_softmax

import copy
import argparse
import time
import os
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, metrics, preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
# 导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import math
from sklearn.metrics import confusion_matrix

from utils.build_dataset import Dataset
from importlib import import_module
# 写一个分batch训练的过程
import itertools
import setproctitle
setproctitle.setproctitle("DGL DEMO")
from tqdm import tqdm
from modules.loss import MultiLabelLoss
import numpy as np
# from dgl.nn.pytorch.conv.sageconv_mc import SAGEConv_MC

# from models.SageconvSelect import SAGEConv_MC



def prepare_text_model(model_name='TextCNN_v3', n_classes=10):
    '''
    '''

    print(f'using model{model_name}')
    config_model = import_module('models.' + model_name)
    config = config_model.Config(n_classes=n_classes)
    model = config_model.TextCNN(config)
    return model



# def train_val_test_mask(M, train_size, split_val=True, load_fix_nid = False, save_nid = False, filter_nodes = []):
def train_val_test_mask(M, train_size, split_val=True, load_fix_nid = False, save_nid = False):
    num_samples = len(M)
    if load_fix_nid == True:
        f=open("save_nid.txt","r")
        nid_list = f.readlines()
        shuffle_idx =[int(nid) for nid in nid_list]
        f.close()
        print("Direct load_fix_nid done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         print(shuffle_idx)
    else:
#         print(M)
        mask_idx_0 = [id for id,m in enumerate(M) if m==0]
        mask_idx_1 = [id for id,m in enumerate(M) if m==1]
    #     print("mask_idx_0, mask_idx_1", len(mask_idx_0), len(mask_idx_1))

        random.shuffle(mask_idx_0)
        random.shuffle(mask_idx_1)
        mask_idx = mask_idx_0 + mask_idx_1
        
#         mask_idx = [idx for idx in mask_idx if idx not in filter_nodes]

        random.shuffle(mask_idx)

        if save_nid == True:
            mask_idx = mask_idx[:1000]

#         mask_idx = mask_idx[:200]
            
        shuffle_idx = mask_idx

#         print("shuffle_idx:",  shuffle_idx)


        if save_nid == True:
            f=open("save_nid.txt","w")
            nid_list =[str(nid)+"\n" for nid in shuffle_idx]
            f.writelines(nid_list)
            f.close()
            print("Re save nid done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         print(shuffle_idx)
    
    num_mask = len(shuffle_idx)

#     print("num_samples, num_mask:", num_samples, num_mask )

    train_split = int(num_mask*train_size)
    train_idx = shuffle_idx[:train_split]
    if split_val:
        val_split = int(num_mask*((1+train_size))/2)
        val_idx = shuffle_idx[train_split:val_split]
        test_idx = shuffle_idx[val_split:]
    else:
        test_idx = shuffle_idx[train_split:]
        
    train_mask = np.zeros(num_samples)
    train_mask[train_idx] = 1
    if split_val:
        val_mask = np.zeros(num_samples)
        val_mask[val_idx] = 1
    test_mask = np.zeros(num_samples)
    test_mask[test_idx] = 1

    if split_val:
        return train_mask, val_mask, test_mask, train_idx, test_idx, val_idx
    else:
        return train_mask,test_mask, train_idx, test_idx

