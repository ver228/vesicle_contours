#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:18:33 2018

@author: avelinojaver
"""
from pathlib import Path
import cv2
import math
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from skimage.draw import circle_perimeter

import tqdm

from extract_best_circles import extract_best_circles, norm_and_smooth
from extract_contours import get_interfaces

#%%

    
#%%
if __name__ == '__main__':
    main_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois')
    fnames = list(main_dir.rglob('*.hdf5'))
    
    #fname = fnames[20]
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100.22Sep2016_17.49.11/roi_24.hdf5'
    fname = '/Volumes/rescomp1/data/Vesicles/roi_cleaned/22_09_16/ves5/ramp100bis.22Sep2016_18.01.55/roi_93.hdf5'
    
    
    df_circle = extract_best_circles(fname)
    
    ini_frame = df_circle['frame_number'].min()
    frame_gg = df_circle.groupby('frame_number')
    
    #%%
    min_seg_size = 25
    
    
    with pd.HDFStore(str(fname), 'r') as fid:
        src = fid.get_node('/mask')
        tot, img_w, img_h = src.shape
        
        rx, ry = np.meshgrid(np.arange(img_w), np.arange(img_h))
        def _get_circle_mask(row):
            return np.sqrt((rx-row['cx'])**2 + (ry-row['cy'])**2) < row['radii']
        
        for tt in [0, tot//4, tot//2, 3*tot//4, 7*tot//8, tot-1]:
            row = frame_gg.get_group(tt).iloc[0]
            mask_r = _get_circle_mask(row)
            
            img = src[tt]
            img_n = norm_and_smooth(img) 
            
            contours_valid = get_interfaces(img_n, mask_r, _debug = True)
            
            