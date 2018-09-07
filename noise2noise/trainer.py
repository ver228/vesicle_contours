#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
from .flow import CroppedFlow, _root_dir
from .models import UNet

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import datetime
import shutil
import tqdm

log_dir_root = _root_dir.parent / 'results' / 'logs'

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)
        
def get_loss(loss_type):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(loss_type)
    
    return criterion

def get_model(model_name):
    if model_name == 'unet':
        model = UNet(n_channels = 1, n_classes = 1)
    else:
        raise ValueError(model_name)
    return model

def train(
        loss_type = 'l1',
        cuda_id = 0,
        batch_size = 8,
        model_name = 'unet',
        lr = 1e-4, 
        weight_decay = 0.0,
        n_epochs = 2000,
        num_workers = 1
        ):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    
    
    gen = CroppedFlow()
    loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    model = get_model(model_name)
    model = model.to(device)
    
    criterion = get_loss(loss_type)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
    bn = '{}_{}_{}_lr{}_wd{}_batch{}'.format(loss_type, bn, 'adam', lr, weight_decay, batch_size)
    log_dir = log_dir_root / bn
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
        #%%
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        gen.train()
        pbar = tqdm.tqdm(loader)
        
        avg_loss = 0
        frac_correct = 0
        for X, target in pbar:
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            avg_loss += loss.item()
            
            
        
        avg_loss /= len(loader)
        frac_correct /= len(gen)
        tb = [('train_epoch_loss', avg_loss)]
        
        for tt, val in tb:
            logger.add_scalar(tt, val, epoch)
            
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))