#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:08:14 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))


from noise2noise.flow import CroppedFlow
from noise2noise.models import UNet

from torch import nn

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm
    
    gen = CroppedFlow()
    loader = DataLoader(gen, batch_size=8)
    
    gen.train()
    tops = []
    bots = []
    
    mod = UNet(n_channels = 1, n_classes = 1)
    for X,Y in tqdm.tqdm(loader):
        Xhat = mod(X)
        break