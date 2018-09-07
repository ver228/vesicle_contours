#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:11:34 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from noise2noise.models import UNet
from noise2noise.trainer import log_dir_root

from read_movies.moviereader import MovieReader

import numpy as np
import torch
import tqdm
import math
from scipy.ndimage.filters import median_filter

import torch.nn.functional as F

if __name__ == '__main__':
    
    model = UNet(n_channels = 1, n_classes = 1)
    
    #model_path = log_dir_root / 'l2_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    model_path = log_dir_root / 'l1_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    movie_name = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/script_ramp.08Dec2015_16.45.56.movie'
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/script_ramp.08Dec2015_17.09.35.movie'

    reader = MovieReader(movie_name)
    
    tot = len(reader)
    #%%
    scale = (7, 11.1)
    
    frame_number = 1066#tot-1#1761
    _, img = reader[frame_number]
    X = (np.log(img)-scale[0])/(scale[1]-scale[0])
    
    X = torch.from_numpy(X)[None, None]
    #%%
    with torch.no_grad():
        Xhat = model(X)
    
    #%%
    x = X.squeeze()
    xhat = Xhat.detach().numpy().squeeze()
    
    xmed_r = median_filter(img, size=5)
    xmed = (np.log(xmed_r)-scale[0])/(scale[1]-scale[0])
    
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    axs[0].imshow(x)
    axs[1].imshow(xhat)
    axs[2].imshow(xmed)
    
    
    #%%