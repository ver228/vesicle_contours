#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:41:53 2018

@author: avelinojaver
"""
from pathlib import Path
import tqdm

from extract_rois import extract_rois
from extract_contours import track_interface
#%%
if __name__ == '__main__':
    movies_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/')
    save_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/')
    
    movie_files = list(movies_root_dir.rglob('*.hdf5'))
    #%%
    for movie_file in tqdm.tqdm(movie_files):
        save_dir = Path(str(movie_file.parent).replace(str(movies_root_dir), str(save_root_dir)))
        extract_rois(movie_file,  
                     save_dir, 
                     min_good_fraction = 0.5, 
                     min_vesicle_radius = 100,
                     ignore_mixed = True)
        
    #%%
    roi_files = list(save_root_dir.rglob('*.hdf5'))
    for roi_file in tqdm.tqdm(roi_files):
        track_interface(roi_file)
    