import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelLoss(nn.Module):
    '''
    inputs: [batchsize,n_class]
    labels: [batchsize,n_class]
    '''

    def __init__(self,pos_weight=None):
        '''
        '''
        super(MultiLabelLoss,self).__init__()

        self.loss_fcn = nn.BCEWithLogitsLoss(pos_weight)

    def forward(self, outputs, labels):
        '''
        '''
        labels = labels.float()
        
        loss = self.loss_fcn(outputs,labels)

        return loss
