#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:25:03 2018

@author: avelinojaver
"""

import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from noise2noise.models import UNet
from noise2noise.trainer import log_dir_root

import cv2
import torch
import tqdm
import numpy as np
import math

root_dir = Path.home() / 'workspace' / 'Vesicles' / 'training_data'
if __name__ == '__main__':
    #%%
    base_batch_size = 16
    base_size = 640
    #%%
    scale = (7, 11.1)
    cuda_id = 0
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = UNet(n_channels = 1, n_classes = 1)
    model_path = log_dir_root / 'l1_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    state = torch.load(str(model_path), map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    model = model.to(device)
    
    dname_src = root_dir / 'initial_frames'
    dname_dst = root_dir / 'cleaned_frames'
    
    
    fnames = list(dname_src.rglob('*.png'))
    #%%
    
    def _process_batch(data):
        if not data:
            return []
        
        new_fnames, imgs = zip(*data)
            
        with torch.no_grad():
            
            X = torch.from_numpy(np.stack(imgs).astype(np.float32))
            X = X.to(device)
            X.log_()
            X -= scale[0]
            X /= (scale[1]-scale[0])
            
            Xnew = model(X)
            
            Xnew *= (scale[1]-scale[0])
            Xnew += scale[0]
            Xnew.exp_()
            
            #ensure this is between the unint16 range
            Xnew.round_().clamp_(min=1, max=65535)
            
            Xnew = Xnew.detach().cpu().numpy()
        
        for nfname, img in zip(new_fnames, Xnew):
            img = img.squeeze().astype(np.uint16)
            
            cv2.imwrite(nfname, img)
            
        return []
        
    batch_data = []
    
    prev_shape = (540, 640)
    batch_size = base_batch_size
    
    for fname in tqdm.tqdm(sorted(fnames)):
        new_name = str(fname).replace(str(dname_src), str(dname_dst))
        if Path(new_name).exists():
            continue
        
        img = cv2.imread(str(fname), -1)
        if img is None:
            continue
        
        Path(new_name).parent.mkdir(exist_ok=True, parents=True)
        
        #process batch if either the batch is complete or the image sizes do not match
        if len(batch_data) >= batch_size or (img.shape != prev_shape):
            batch_data = _process_batch(batch_data)
            
            #adjust the batch size according to the image size
            batch_size = max(1, math.floor(base_batch_size*base_size**2/max(img.shape)**2))
            
            
        
        prev_shape = img.shape
        
        
        
        batch_data.append((new_name, img[None]))
        
    batch_data = _process_batch(batch_data)
        
        
        