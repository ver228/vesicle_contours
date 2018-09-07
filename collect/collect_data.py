#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:03:41 2018

@author: avelinojaver
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'read_movies'))

from moviereader import MovieReader
import tqdm

import cv2
import pandas as pd
import math
import random
#%%
if __name__ == '__main__':
    save_dir_root = Path('/Volumes/rescomp1/data/Vesicles/training_data/initial_frames')
    
    base_dir = Path('/Volumes/Ext1/Data/AnalysisFingers/')
    all_files = list(base_dir.rglob('*.movie'))
    
    tot = len(all_files)
    test_frac = 0.05
    is_train = math.ceil(tot*test_frac)*[False] + math.floor(tot*(1-test_frac))*[True]
    random.shuffle(is_train)
    
    #%%
    img2sample = 5
    
    files_data = []
    
    ifname = 0
    for fname in tqdm.tqdm(all_files):
        mreader = MovieReader(fname)
        if len(mreader) <= img2sample + 1:
            continue
        
        ifname += 1
        files_data.append((ifname, str(fname).replace(str(base_dir), '')))
        
        s_type = 'train' if is_train[ifname] else 'test'
        save_dir = save_dir_root / s_type / str(ifname)
        save_dir.mkdir(exist_ok=True, parents=True)
            
        for ii, (header, img) in enumerate(mreader):
            save_name = save_dir / '{}_{}.png'.format(ifname, ii)
            cv2.imwrite(str(save_name), img)
            
            if ii >= img2sample + 1:
                break
        
    #%%
    save_name = str(save_dir_root / 'file_names.csv')
    df = pd.DataFrame(files_data, columns=['id', 'path'])
    df.to_csv(save_name, index=False)
