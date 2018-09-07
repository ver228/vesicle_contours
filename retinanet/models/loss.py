#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:38:56 2018

@author: avelinojaver
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_embedding(labels, num_classes):
    y = torch.eye(3)
    return y[labels]
    

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = 0.25, gamma = 2.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, preds, targets):
        target_onehot = torch.eye(self.num_classes+1)[targets]
        target_onehot = target_onehot[:,1:].contiguous() #the zero is the background class
        target_onehot = target_onehot.to(targets.device) #send to gpu if necessary
        
        focal_weights = self._get_weight(preds,target_onehot)
        
        #I already applied the sigmoid to the classification layer. I do not need binary_cross_entropy_with_logits
        return (focal_weights*F.binary_cross_entropy(preds, target_onehot, reduce=False)).sum()
    
    def _get_weight(self, x, t):
        pt = x*t + (1-x)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)
    
    def forward(self, pred, target):
        #%%
        clf_target, loc_target = target
        clf_preds, loc_preds = pred
        
        ### regression loss
        pos = clf_target > 0
        num_pos = pos.sum().item()
        
        #since_average true is equal to divide by the number of possitives
        loc_loss = F.smooth_l1_loss(loc_preds[pos], loc_target[pos], size_average=False)
        loc_loss = loc_loss/max(1, num_pos)
        
        #### focal lost
        valid = clf_target >= 0  # exclude ambigous anchors (jaccard >0.4 & <0.5) labelled as -1
        clf_loss = self.focal_loss(clf_preds[valid], clf_target[valid])
        clf_loss = clf_loss/max(1, num_pos)  #inplace operations are not permitted for gradients
        
        
        #I am returning both losses because I want to plot them separately
        return clf_loss, loc_loss
