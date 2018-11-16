#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:05:32 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[0]
sys.path.append(str(root_dir))

from extract_rois import extract_rois
from extract_contours import track_interface

import tqdm

movies_root_dir = Path.home() / 'workspace/Vesicles/movies_cleaned'
save_root_dir = Path.home() / 'workspace/Vesicles/roi_cleaned'


def process_file(movie_file):
    movie_file = Path(movie_file)
    
    save_dir_r = Path(str(movie_file.parent).replace(str(movies_root_dir), str(save_root_dir)))
    save_dir = save_dir_r / movie_file.stem
    
    extract_rois(movie_file,  
                 save_dir, 
                 min_good_fraction = 0.5, 
                 min_vesicle_radius = 100,
                 ignore_mixed = True)
    
    
    if not save_dir.exists():
        return
    
    roi_files = list(save_dir.rglob('*.hdf5'))
    for roi_file in tqdm.tqdm(roi_files):
        track_interface(roi_file)
#%%
if __name__ == '__main__':
    process_file(sys.argv[1])
    
    #%%
    #fname = Path.home() / 'workspace/Vesicles/movies_cleaned/RAMPSfingers20_10_15/movieAsameves_180gl_25to45_70.20Oct2015_16.41.53.hdf5'
    #fname = Path.home() / 'workspace/Vesicles/movies_cleaned/22_09_16/ves5/ramp100bis.22Sep2016_18.01.55.hdf5'
    #process_file(fname)
    