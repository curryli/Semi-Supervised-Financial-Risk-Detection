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
from sklearn.metrics import pairwise
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import math
from sklearn.metrics import confusion_matrix
from utils.build_dataset import Dataset, DatasetGT
from importlib import import_module
import itertools
import setproctitle
setproctitle.setproctitle("DGL DEMO")
from tqdm import tqdm
from modules.loss import MultiLabelLoss
import numpy as np
from models.SageconvDualAttWeight import SAGEConv_DA
from util_share import prepare_text_model,train_val_test_mask
import threading
import time

class GraphSAGE_cross(nn.Module):
    def __init__(self, 
                 text_feat_size,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator,
                label_rela_matrix):

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.feat_drop = 0.0
        self.label_rela_matrix = label_rela_matrix
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv_DA(text_feat_size, n_hidden, aggregator, self.feat_drop, self.label_rela_matrix))
        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv_DA(n_hidden, n_hidden, aggregator, self.feat_drop, self.label_rela_matrix))
        self.layer.append(SAGEConv_DA(n_hidden, n_classes, aggregator, self.feat_drop, self.label_rela_matrix))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def load_subtensor(self, nfeat, input_nodes, device, text_pred_labels, output_nodes, ori_labels):
        batch_input_feats = nfeat[input_nodes].to(device)
        batch_input_labels_text = text_pred_labels[input_nodes].to(device)
        batch_input_labels_ori = ori_labels[input_nodes].to(device)
        
        batch_cent_feats = nfeat[output_nodes].to(device)
        batch_cent_labels_text = text_pred_labels[output_nodes].to(device)
        batch_cent_labels_ori = ori_labels[output_nodes].to(device)

        return batch_input_feats, batch_input_labels_text, batch_input_labels_ori, batch_cent_feats, batch_cent_labels_text, batch_cent_labels_ori
   
    def forward(self, blocks, batch_args, device):
        batch_input_feats, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori = batch_args
        h = self.dropout(batch_input_feats)

        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            layer_args = h, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori
            h, batch_input_labels, batch_input_labels_ori = layer(block, layer_args)

            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h   

    def inference(self, g, nfeat, text_pred_labels, ori_labels, batch_s, num_worker, device, val_nid, inf_feat_flag=False):
        by = torch.zeros(g.num_nodes(), self.n_classes)
        by_ori = torch.zeros(g.num_nodes(),  self.n_classes)
        for l, layer in enumerate(self.layer):
            if inf_feat_flag==True and l==len(self.layer)-2:
                return nfeat
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layer) - 1 else self.n_classes)
            
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    val_nid,
                    sampler,
                    batch_size = batch_s,
                    shuffle=True,
                    drop_last=False,
                    num_workers=num_worker
                )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                block = block.int().to(device)
                if l==0:
                    batch_input_feats, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori = self.load_subtensor(nfeat, input_nodes, device, text_pred_labels, output_nodes, ori_labels)
                else:
                    batch_input_feats = nfeat[input_nodes].to(device)
                    batch_input_labels = by[input_nodes].to(device)
                    batch_input_labels_ori = by_ori[input_nodes].to(device)

                h = batch_input_feats
                layer_args = h, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori
                h, batch_input_labels,batch_input_labels_ori = layer(block, layer_args)
                if l != self.n_layers - 1 :
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.cpu()
                by[output_nodes] = batch_input_labels.float().cpu()
                by_ori[output_nodes] = batch_input_labels_ori.float().cpu()
            nfeat = y
            
        return y

def evaluate(model, g, nfeat, text_pred_labels, ori_labels, val_nid, val_mask, batch_s, num_worker, device):
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(g, nfeat, text_pred_labels, ori_labels, batch_s, num_worker, device, val_nid)

    label_pred = torch.sigmoid(label_pred)
    predicted_label = (label_pred.data>0.5).long().cpu().numpy()#.flatten()

    acc = metrics.accuracy_score(predicted_label[val_mask], ori_labels[val_mask].long().cpu().numpy())
    
    report  = metrics.classification_report(predicted_label[val_mask], ori_labels[val_mask].long().cpu().numpy(), digits=6, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    micro_f1 = report['micro avg']['f1-score']
    
    return acc,macro_f1,micro_f1



def get_inf_feat(model, g, nfeat, text_pred_labels, ori_labels, val_nid, val_mask, batch_s, num_worker, device):
    model.eval()
    with torch.no_grad():
        feat_pred = model.inference(g, nfeat, text_pred_labels, ori_labels, batch_s, num_worker, device, val_nid, inf_feat_flag=True)
    return feat_pred


def get_window_max(arr_list, win_size):
    win_max = 0.0
    for i in range(len(arr_list)-win_size+1):
        tmp_arr = arr_list[i:i+win_size]
        tmp_max = np.mean(tmp_arr)
        if tmp_max>win_max:
            win_max = tmp_max
    return win_max

def train_graph_cross(g, X, Y, M, args_split, args_text_pred, args, graph_model):
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker, n_classes, device, lr_graph, lr_text, text_state_dict_path, graph_state_dict_path, text_epoch, graph_epoch, preload_text, preload_graph, sample_size, load_fix_nid, save_nid = args
    train_mask, val_mask, test_mask, train_nid, test_nid, val_nid, X_train, Y_train, X_val, Y_val, X_test, Y_test = args_split
    text_feat_size, text_pred_feat, text_pred_labels, ori_labels = args_text_pred
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size = batch_s,
        shuffle=True,
        drop_last=False,
        num_workers=num_worker
    )
    
    optimizer = torch.optim.Adam(graph_model.parameters(), lr=lr_graph)
    loss_fun = MultiLabelLoss()
    loss_fun.to(device)

    acc_list = []
    microf1_list = []
    macrof1_list = []
    for epoch in range(graph_epoch):
        graph_model.train()
        print(f"****************************Graph Epoch{epoch}****************************")
        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):

            batch_input_feats, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori = graph_model.load_subtensor(text_pred_feat, input_nodes, device, text_pred_labels, output_nodes, ori_labels)
            block = [block_.int().to(device) for block_ in block]

            batch_args = batch_input_feats, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori
            
            graph_model_pred = graph_model(block, batch_args, device)
            loss = loss_fun(graph_model_pred, batch_cent_labels_ori)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                print(graph_model_pred.shape)
                print('Batch %d | Loss: %.8f' % (batch, loss.item()))

        if epoch % 1 == 0:
            print(f"________________Graph Test epoch{epoch}:_______________")
            test_acc, macro_f1, micro_f1 = evaluate(graph_model, g, text_pred_feat, text_pred_labels, ori_labels, test_nid, test_mask, batch_s, num_worker, device)
            print("macro_f1, micro_f1, test_acc",macro_f1, micro_f1, test_acc)
            macrof1_list.append(macro_f1)
            microf1_list.append(micro_f1)
            acc_list.append(test_acc)

    window_max_macro = get_window_max(macrof1_list, 5)
    window_max_micro = get_window_max(microf1_list, 5)
    window_max_acc = get_window_max(acc_list, 5)
    print("Best Graph macro f1 in last 5 avg:", window_max_macro)
    print("Best Graph micro f1 in last 5 avg:", window_max_micro)
    print("Best Graph Acc in last 5 avg:", window_max_acc)
        
    feat_pred = get_inf_feat(graph_model, g, text_pred_feat, text_pred_labels, ori_labels, val_nid, val_mask, batch_s, num_worker, device)
    return feat_pred


def train_text(args_split, args, text_model, Graph_emb=None):
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker, n_classes, device, lr_graph, lr_text, text_state_dict_path, graph_state_dict_path, text_epoch, graph_epoch, preload_text, preload_graph, sample_size, load_fix_nid, save_nid = args
    train_mask, val_mask, test_mask, train_nid, test_nid, val_nid, X_train, Y_train, X_val, Y_val, X_test, Y_test = args_split
    
    batch_text = 4
    dataloader_params = {'batch_size': batch_s,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory':True}
    
    text_optimizer = torch.optim.Adam(text_model.parameters(), lr=lr_text)
    text_loss_fun = MultiLabelLoss()
    text_loss_fun.to(device)
    
    print("Train Text：：：：：：：：：：：：:")
    acc_list = []
    microf1_list = []
    macrof1_list = []
    if Graph_emb is None:
        train_dataset = Dataset(X_train,Y_train)
        training_generator = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
        test_dataset = Dataset(X_test,Y_test)
        testing_generator = torch.utils.data.DataLoader(test_dataset, **dataloader_params)

        best_f1=0.0
        for epoch in range(text_epoch):
            text_model.train() 
            total_loss = 0
            for i, (trains, labels) in enumerate(training_generator):
                trains, labels = trains.to(device), labels.to(device)
                outputs = text_model(trains)
                text_model.zero_grad()
                loss = text_loss_fun(outputs, labels)
                total_loss += loss.data.item()
                loss.backward()
                text_optimizer.step()

            train_loss = total_loss / len(training_generator)
            print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS" % (epoch, train_loss, math.exp(train_loss)))

            # test step
            if epoch%1==0:
                text_model.eval()
                predicted_label = []
                groud_truth = []
                with torch.no_grad():
                    for texts, labels in testing_generator:
                        texts, labels = texts.to(device), labels.to(device)
                        labels = labels.data.cpu().numpy()
                        groud_truth.append(labels)
                        outputs = text_model(texts)
                        outputs = torch.sigmoid(outputs)
                        predicted = (outputs.data>0.5).long().cpu().numpy()
                        predicted_label.append(predicted)

                labels_all = np.vstack(groud_truth)#.astype(np.int32)
                predict_all = np.vstack(predicted_label)#.astype(np.int32)
                acc = metrics.accuracy_score(labels_all, predict_all)
                print(f"****************************Text Epoch{epoch}****************************")
                print(f'Test Acc: {acc:.4f}')
                report = metrics.classification_report(labels_all, predict_all, digits=6, output_dict=True)

                macro_f1 = report['macro avg']['f1-score']
                micro_f1 = report['micro avg']['f1-score']
                print("macro_f1, micro_f1, acc",macro_f1, micro_f1, acc)
            
                macrof1_list.append(macro_f1)
                microf1_list.append(micro_f1)
                acc_list.append(acc)

                if epoch>10 and macro_f1>best_f1:
                    best_f1 = macro_f1
                    
    else:
        G_emb_np = Graph_emb.cpu().numpy()
        G_train = G_emb_np[train_nid]
        G_val =  G_emb_np[val_nid]
        G_test = G_emb_np[test_nid]
        
        train_dataset = DatasetGT(X_train,Y_train,G_train)
        training_generator = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
        test_dataset = DatasetGT(X_test,Y_test,G_test)
        testing_generator = torch.utils.data.DataLoader(test_dataset, **dataloader_params)

        best_f1=0.0
        for epoch in range(text_epoch):
            text_model.train() 
            total_loss = 0
            for i, (trains, labels, g_embs) in enumerate(training_generator):
                trains, labels, g_embs = trains.to(device), labels.to(device), g_embs.to(device)
                outputs = text_model(trains, g_embs)
                ##############################
                text_model.zero_grad()
                loss = text_loss_fun(outputs, labels)
                total_loss += loss.data.item()
                loss.backward()
                text_optimizer.step()

            train_loss = total_loss / len(training_generator)
            # test step
            if epoch%1==0:
                text_model.eval()
                predicted_label = []
                groud_truth = []
                with torch.no_grad():
                    for texts, labels, g_embs in testing_generator:
                        texts, labels, g_embs = texts.to(device), labels.to(device), g_embs.to(device)
                        labels = labels.data.cpu().numpy()
                        groud_truth.append(labels)
                        outputs = text_model(texts,g_embs)
                        outputs = torch.sigmoid(outputs)
                        predicted = (outputs.data>0.5).long().cpu().numpy()
                        predicted_label.append(predicted)

                labels_all = np.vstack(groud_truth)#.astype(np.int32)
                predict_all = np.vstack(predicted_label)#.astype(np.int32)

                acc = metrics.accuracy_score(labels_all, predict_all)
                print(f"****************************Text Epoch{epoch}****************************")
                print(f'Test Acc: {acc:.4f}')
                report  = metrics.classification_report(labels_all, predict_all, digits=6, output_dict=True)
                macro_f1 = report['macro avg']['f1-score']
                micro_f1 = report['micro avg']['f1-score']
        
                macrof1_list.append(macro_f1)
                microf1_list.append(micro_f1)
                acc_list.append(acc)
                print("macro_f1, micro_f1, acc",macro_f1, micro_f1, acc)
            
                if epoch>10 and macro_f1>best_f1:
                    best_f1 = macro_f1

    window_max_macro = get_window_max(macrof1_list, 5)
    window_max_micro = get_window_max(microf1_list, 5)
    window_max_acc = get_window_max(acc_list, 5)
     
    print("Update Text model.....")
    torch.save(text_model.state_dict(), text_state_dict_path)
    print(f"Save Text model {text_state_dict_path} done!!!!")
    return None

def predict_text_tensor(X, Y, text_model, test_nid, val_nid, device, Graph_emb=None):   
    print("Use New Text model to inference Text Tensor and label:")
    feat_tensor = None
    label_tensor = None
    text_model.eval()
    with torch.no_grad():
        
        if Graph_emb is not None:
            for i in tqdm(range(len(X))):
                batch_ids = X[i]
                batch_ids = torch.unsqueeze(torch.LongTensor(batch_ids).to(device), 0)
                batch_g = torch.unsqueeze(Graph_emb[i], 0).to(device)

                batch_input_feats = text_model.text_cnn_embedding_infer(batch_ids, batch_g)
                batch_input_labels = text_model(batch_ids, batch_g)

                if i==0:
                    feat_tensor = batch_input_feats
                    label_tensor = batch_input_labels
                else:
                    feat_tensor = torch.cat((feat_tensor, batch_input_feats), 0)
                    label_tensor = torch.cat((label_tensor, batch_input_labels), 0)
        else:
            for i in tqdm(range(len(X))):
                batch_ids = X[i]
                batch_ids = torch.unsqueeze(torch.LongTensor(batch_ids).to(device), 0)

                batch_input_feats = text_model.text_cnn_embedding_infer(batch_ids)
                batch_input_labels = text_model(batch_ids)

                if i==0:
                    feat_tensor = batch_input_feats
                    label_tensor = batch_input_labels
                else:
                    feat_tensor = torch.cat((feat_tensor, batch_input_feats), 0)
                    label_tensor = torch.cat((label_tensor, batch_input_labels), 0) 

    text_feat_size = feat_tensor.shape[1]
    text_pred_feat = feat_tensor
    outputs = label_tensor
    outputs = torch.sigmoid(outputs)
    text_pred_labels = (outputs.data>0.8).long()
    ori_labels = torch.LongTensor(Y)

    return text_feat_size, text_pred_feat, text_pred_labels, ori_labels


def train_pro(g, X, Y, M, train_val_test_mask, args):
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker, n_classes, device, lr_graph, lr_text, text_state_dict_path, graph_state_dict_path, text_epoch, graph_epoch, preload_text, preload_graph, sample_size, load_fix_nid, save_nid = args

    train_mask, val_mask, test_mask, train_nid, test_nid, val_nid  = train_val_test_mask(M, train_size=0.6, split_val=True, load_fix_nid = load_fix_nid, save_nid = save_nid)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    
    text_model = prepare_text_model('TextModel', n_classes)
    text_model.to(device)
    
    X_train = np.array(X)[train_nid]
    Y_train = np.array(Y)[train_nid]
    
    X_val = np.array(X)[val_nid]
    Y_val = np.array(Y)[val_nid]
    
    X_test = np.array(X)[test_nid]
    Y_test = np.array(Y)[test_nid]
    
    label_rela_matrix = pairwise.cosine_similarity(Y_train.T)

    args_split = train_mask, val_mask, test_mask, train_nid, test_nid, val_nid, X_train, Y_train, X_val, Y_val, X_test, Y_test

    text_feat_size = 768
    if preload_text==True:
        text_params = torch.load(text_state_dict_path)    
        print("Text model loaded done!!!!")
        text_model.load_state_dict(text_params)
         
    # print("第0轮，未传入Graph_emb训练CNNEncoder")
    train_text(args_split, args, text_model)

    # print("加载该轮最优的text参数!!!!")
    text_params = torch.load(text_state_dict_path)    
    text_model.load_state_dict(text_params)

    print("用新一轮的Text model inference 特征 done!!!!")
    text_feat_size, text_pred_feat, text_pred_labels, ori_labels = predict_text_tensor(X, Y, text_model, test_nid, val_nid, device) 
    args_text_pred = text_feat_size, text_pred_feat, text_pred_labels, ori_labels        
            
    ############################################训练初始graph model#########################################
    graph_model = GraphSAGE_cross(text_feat_size, hidden_size, n_classes, n_layers, activation, dropout, aggregator, label_rela_matrix)
    graph_model.to(device)
        
    print(graph_model)
    Graph_emb = train_graph_cross(g, X, Y, M, args_split, args_text_pred, args, graph_model)  #torch.Size([8876, 768])
    torch.save(graph_model.state_dict(), graph_state_dict_path)
    print("Graph model train done!!!!")
        
    for i in range(10):
        print(f"大迭代轮数：{i}\n")
        #############利用中心节点训练 Text模型，并inference节点#########################################
        print("传入Graph_emb训练Text model")  #text_model自动使用上一轮的text_model，不用重新加载
        train_text(args_split, args, text_model, Graph_emb)

        print("用新一轮的Text model inference 特征 done!!!!")
        text_feat_size, text_pred_feat, text_pred_labels, ori_labels = predict_text_tensor(X, Y, text_model, test_nid, val_nid, device, Graph_emb)

        text_params = torch.load(text_state_dict_path)    
        text_model.load_state_dict(text_params)
        
        args_text_pred = text_feat_size, text_pred_feat, text_pred_labels, ori_labels
        
        ########################训练Graph Model##################################################
        graph_params = torch.load(graph_state_dict_path)    
        graph_model.load_state_dict(graph_params)
        print("graph model loaded done!!!!")
        Graph_emb = train_graph_cross(g, X, Y, M, args_split, args_text_pred, args, graph_model) 
        print("Graph_emb.shape", Graph_emb.shape)
        
        torch.save(graph_model.state_dict(), graph_state_dict_path)
        print("Graph model train done!!!!")

            

            


