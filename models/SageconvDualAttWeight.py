# -*- coding: utf-8 -*-
"""Torch Module for GraphSAGE layer"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import dgl.function as fn



def expand_as_pair(input_, g=None):
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()}
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

def check_eq_shape(input_):
    srcdata, dstdata = expand_as_pair(input_)
    src_feat_shape = tuple(F.shape(srcdata))[1:]
    dst_feat_shape = tuple(F.shape(dstdata))[1:]
    if src_feat_shape != dst_feat_shape:
        print("The feature shape of source nodes: {} \
            should be equal to the feature shape of destination \
            nodes: {}.".format(src_feat_shape, dst_feat_shape))


class SAGEConv_DA(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop,
                 label_rela_matrix,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv_DA, self).__init__()

#         print("SAGEConv_MC in_feats", in_feats)

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
#         self.norm = torch.nn.functional.normalize
        self.norm = norm
        self.activation = activation #F.relu
        self.feat_drop = nn.Dropout(feat_drop)

        self.label_rela_matrix = label_rela_matrix
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

#         print("SAGEConv_MC self._in_src_feats, self._in_dst_feats", self._in_src_feats, self._in_dst_feats)

#         print("self._in_src_feats, out_feats", self._in_src_feats, out_feats)
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)

#         self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)

        self.fc_half = nn.Linear(out_feats*2, out_feats)
        
        init_eps=0.5
        self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)

        self.num_heads = 8
 
        self.fc_trans_attn = nn.Linear(self._out_feats*2, self._out_feats * self.num_heads)

        self.feat_attn_W = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self._out_feats)))
#         self.feat_attn_W = nn.Parameter(torch.FloatTensor(size=(1, self._in_src_feats, self._in_src_feats)))  #e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            
        gain = 1.0

#         print(self.fc_self.weight)


        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_half.weight, gain=gain)

            nn.init.xavier_uniform_(self.fc_trans_attn.weight, gain=gain)
            nn.init.xavier_uniform_(self.feat_attn_W, gain=gain)


    def _select_mean(self, nodes):
        cent_l = nodes.data['l'] #(Batch_size, 10维)
        cent_f = nodes.data['f'] #(Batch_size, 10维)

#         print("cent_l.shape", cent_l.shape)

        m = nodes.mailbox['m'] # (Batch_size, 邻居节点个数, 特征维度)
        
        batch_size = m.shape[0]
        nei_size =  m.shape[1]
        h = m[:, :, :-20]
        l_pred = m[:, :, -20:-10]
        l_ori = m[:, :, -10:]
        
        
        device = cent_l.get_device()
#         print("cent_l.get_device()", device)

#         print(l_pred)
        cent_l_arr = cent_l.cpu().numpy()
        cent_f_arr = cent_f.cpu().detach().numpy()
    
        l_pred_arr = l_pred.cpu().detach().numpy()
        h_arr = h.cpu().detach().numpy()
        
        
        h_activate_tensor = self.leaky_relu(h)   ## (64, 3, 768)  (batch_size, neighbor_size, feat_size)
   
        batch_label_sim_tensor = torch.ones(batch_size, nei_size).to(device)
        batch_feat_sim_tensor = torch.ones(batch_size, nei_size).to(device)
        batch_sim_merge_tensor = torch.ones(batch_size, nei_size).to(device)
        
        for i in range(len(cent_l_arr)):
            def get_multilabel_sim(x): 
                return np.sum(np.outer(x, cent_l_arr[i])*self.label_rela_matrix)

            
            batch_label_sim = map(get_multilabel_sim, l_pred_arr[i])
            batch_label_sim_tensor[i] = torch.Tensor(list(batch_label_sim)).to(device)

            cent_f_i_repeat = cent_f[i].repeat(h_activate_tensor[i].shape[0],1)

            tmp_h_tensor = self.fc_trans_attn(torch.cat([h_activate_tensor[i], cent_f_i_repeat], dim=1)).view(-1, self.num_heads, self._out_feats)
						
            tmp_h_tensor = tmp_h_tensor*self.feat_attn_W
            att_feat_e = tmp_h_tensor.sum(dim=-1).sum(dim=-1)
            batch_feat_sim_tensor[i] = att_feat_e.to(device)
																				
            batch_sim_merge_tensor[i] = batch_label_sim_tensor[i]*self.eps + batch_feat_sim_tensor[i]*(1-self.eps)
																 
						
        Softmax = nn.Softmax(dim=1)
        batch_att = Softmax(torch.tensor(batch_sim_merge_tensor, dtype=torch.float))
        
        batch_att = batch_att
#         h = h.to(device)
        b_expand = batch_att.unsqueeze(2).repeat(1,1,h.shape[2])


        h_att_sum = torch.mul(h, b_expand)
        rst = h_att_sum.mean(1)
        rst = rst.to(torch.float32)
        return {'neigh': rst}

    def forward(self, graph, layer_args):
        batch_input_feats, batch_input_labels, batch_input_labels_ori, batch_cent_feats, batch_cent_labels, batch_cent_labels_ori = layer_args

        with graph.local_scope():
          
            feat_src = feat_dst = self.feat_drop(batch_input_feats)
            label_src = label_dst = batch_input_labels
            label_ori_src = label_ori_dst = batch_input_labels_ori

            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                label_dst = label_src[:graph.number_of_dst_nodes()]
                label_ori_dst = label_ori_src[:graph.number_of_dst_nodes()]
            

            msg_fn = fn.copy_src('h', 'm')


            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
                        # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                    
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
                
            elif self._aggre_type == 'select_mean':
                temp_h = self.fc_neigh(feat_src) if lin_before_mp else feat_src

                graph.srcdata['h'] = torch.cat([temp_h, label_src, label_ori_src], dim=1)

#                 print("graph.srcdata['h']", graph.srcdata['h'].shape)
#                 print("graph.srcdata['h']", graph.srcdata['h'])

                graph.dstdata['l'] = label_dst  #    label_ori_dst #label_dst
                graph.dstdata['f'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                
                graph.update_all(msg_fn, self._select_mean)
                
                h_neigh = graph.dstdata['neigh']
#                 print("h_neigh", h_neigh.shape)
#                 print("lin_before_mp", lin_before_mp)
                    
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))


#             print("h_self", h_self)
#             print("h_neigh", h_neigh)
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = (self.fc_self(h_self) + h_neigh)/2


            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst, label_dst, label_ori_dst
