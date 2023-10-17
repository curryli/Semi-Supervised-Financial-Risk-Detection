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

# from pt_modules.graphsage import SAGEModel
# from pt_modules.loss import F1_Loss, BCE_Loss, ROC_Loss, Regression_Loss, DiceBCELoss

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

from utils.build_dataset import load_sigir_pandas_multi_select
# from utils.build_dataset_smi import *

from importlib import import_module
# 写一个分batch训练的过程
import itertools
import setproctitle
setproctitle.setproctitle("DGL DEMO")
from tqdm import tqdm
from modules.loss import MultiLabelLoss
import numpy as np
from util_share import prepare_text_model,train_val_test_mask
from util_flow import train_pro as train_pro_cross

if __name__ == '__main__':

    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    glist, _ = load_graphs(f'data/dglgraphs/merge_graph_sigir')
    g = glist[0]

    g = dgl.remove_self_loop(g)
    src_, dst_ = g.edges()
    g.add_edges(dst_,src_)
    g = dgl.add_self_loop(g)

    print("process graph done.........................................")
    print(g)
    print("load graph done...")

    std_size =2000
    X,Y,M = load_sigir_pandas_multi_select('data/ndatas/node_feat_processed.csv', std_size=std_size, sep='\t')

    n_classes = 10
    hidden_size = 768
    n_layers = 3
    sample_size = [ -1,-1,-1 ]
    
    activation = F.relu
    dropout = 0.5
    aggregator = 'select_mean'   # mean/gcn/pool/lstm /select_mean
    batch_s = 4
    num_worker = 0
    lr_graph = 0.0005
    lr_text =  0.0003
    device = 0
    
    text_epoch = 50
    graph_epoch = 100#10
    
    preload_text = False  #   True
    preload_graph = False
    load_fix_nid = False
    save_nid = False

    
    text_state_dict_path = f'model_state_dict/t405.pt' 
    graph_state_dict_path = f'model_state_dict/g405.pt'

    cuda_str = 'cuda:' + str(device)
    if device >= 0:
        device = torch.device(cuda_str)
    else:
        device = torch.device('cpu')
        
    args = hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker, n_classes, device, lr_graph, lr_text, text_state_dict_path, graph_state_dict_path, text_epoch, graph_epoch, preload_text, preload_graph, sample_size, load_fix_nid, save_nid

    train_pro_cross(g, X, Y, M, train_val_test_mask, args)
