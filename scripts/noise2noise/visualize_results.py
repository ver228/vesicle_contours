#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:08:14 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from noise2noise.flow import CroppedFlow
from noise2noise.models import UNet
from noise2noise.trainer import log_dir_root

import torch
from torch import nn

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm
    
    gen = CroppedFlow()
    gen.test()
    loader = DataLoader(gen, batch_size=1)
    model = UNet(n_channels = 1, n_classes = 1)
    
    #model_path = log_dir_root / 'l2_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    model_path = log_dir_root / 'l1_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    #%%
    for X,Y in tqdm.tqdm(loader):
        with torch.no_grad():
            Xhat = model(X)
        
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        axs[0].imshow(X.squeeze())
        axs[1].imshow(Xhat.detach().numpy().squeeze())
        break
        