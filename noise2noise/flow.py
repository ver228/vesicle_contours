#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:08:14 2018

@author: avelinojaver
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset 

import cv2

import numpy as np
import random


_root_dir = Path.home() / 'workspace' / 'Vesicles' / 'training_data'

class CroppedFlow(Dataset):
    def __init__(self, 
                 root_dir=_root_dir,
                 files2sample = 5,
                 cropping_size = 256,
                 scale = (7, 11.1),
                 expand_factor = 25
                 ):
        
        self.expand_factor = expand_factor
        self.files2sample = files2sample
        self.cropping_size = cropping_size
        self.scale = scale
        
        root_dir = Path(root_dir)  / 'initial_frames'
        self.train_dirs = [x for x in (root_dir / 'train').iterdir() if x.is_dir()]
        self.test_dirs = [x for x in (root_dir / 'test').iterdir() if x.is_dir()]
        
        self.train()
        
    def train(self):
        self.dirs = [x for _ in range(self.expand_factor) for x in self.train_dirs]
        random.shuffle(self.dirs)
        
    def test(self):
        self.dirs = self.test_dirs
    
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, ind):
        dname = self.dirs[ind]
        i_movie = dname.name
        i_img1 = random.randint(0, self.files2sample)
        i_img2 = i_img1+1
        
        #randomize if the previous is going to predict the after or the after the previous
        if random.random() < 0.5:
            i_img2, i_img1 = i_img1, i_img2
        
        
        fname = dname / '{}_{}.png'.format(i_movie, i_img1)
        X = cv2.imread(str(fname), -1)
        
        fname = dname / '{}_{}.png'.format(i_movie, i_img2)
        Y = cv2.imread(str(fname), -1)
        
        #random cropping
        w,h = X.shape
        ix = random.randint(0, w-self.cropping_size)
        iy = random.randint(0, h-self.cropping_size)
        X = X[ix:ix+self.cropping_size, iy:iy+self.cropping_size]
        Y = Y[ix:ix+self.cropping_size, iy:iy+self.cropping_size]
        
        #horizontal flipping
        if random.random() < 0.5:
            X = X[::-1]
            Y = Y[::-1]
        
        #vertical flipping
        if random.random() < 0.5:
            X = X[:, ::-1]
            Y = Y[:, ::-1]
        
        X = np.log(X)
        Y = np.log(Y)
        X = (X-self.scale[0])/(self.scale[1]-self.scale[0])
        Y = (Y-self.scale[0])/(self.scale[1]-self.scale[0])
            
        return X[None], Y[None]
        
        
#%%
if __name__ == '__main__':
    import tqdm
    gen = CroppedFlow()
    loader = DataLoader(gen, batch_size=8, shuffle=True)
    
    gen.train()
    tops = []
    bots = []
    for X,Y in tqdm.tqdm(loader):
        tops.append(max(X.max().item(), Y.max().item()))
        bots.append(min(X.min().item(), Y.min().item()))