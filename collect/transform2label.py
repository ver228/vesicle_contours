#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:21:01 2018

@author: avelinojaver
"""
import numpy as np
import cv2
import tqdm
from pathlib import Path
#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'Vesicles/training_data/'
    frames_root = root_dir / 'initial_frames'
    
    save_root = root_dir / 'dat2label'
    train_root = save_root / 'train'
    train_root.mkdir(parents=True, exist_ok=True)
    
    test_root = save_root / 'test'
    test_root.mkdir(parents=True, exist_ok=True)
    #%%
    
    
    fnames = list(frames_root.rglob('*_0.png'))
    #%%
    for fname in tqdm.tqdm(fnames):
        
        img = cv2.imread(str(fname), -1)
        if img is None:
            continue
        
        img_l = np.log(img+1)
        
        bot, top = np.min(img_l), np.max(img_l)
        img_n = (img_l-bot)/(top-bot)*255
        img_n = img_n.astype(np.uint8)
        
        if 'train' in str(fname.parent):
            save_dir = train_root
        else:
            save_dir = test_root
        
        bn = fname.name.replace('_0.','.')
        
        save_name = save_dir / bn
        cv2.imwrite(str(save_name), img_n)
        
        
    
    