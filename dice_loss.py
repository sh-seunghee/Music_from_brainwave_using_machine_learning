'''
Author: Guangyu Shen
Last modified: 11/30/2018

This module has following componments:

    1. Define dice_loss function
'''


import torch
import numpy as np
import torch.nn.functional as F
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    
    smooth = 1.
    '''
    target = target.type(torch.LongTensor)
    batch_size,_,H,W = target.size()
    target_one_hot = torch.zeros(batch_size,2,H,W).scatter_(1,target,1)
    target_one_hot = target_one_hot.type(Tensor)
    '''
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

