#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:03:14 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))
from read_movies.moviereader import MovieReader

#%%
from extract_rois import filter_trajectories
from extract_best_circles import norm_and_smooth

from scipy.signal import medfilt
import pandas as pd
import tables
import math
import cv2
import tqdm
#%%
if __name__ == '__main__':
    
    median_filter_size = 25
    min_size = median_filter_size*2
    #%%
    
    cleaned_dir = Path.home() / 'workspace/Vesicles/movies_cleaned'
    raw_dir = Path.home() / 'workspace/Vesicles/data'
    
    save_root_dir = Path.home() / 'workspace/Vesicles/segmented/'
    
    
    cleaned_files = list(cleaned_dir.rglob('*.hdf5'))
    raw_files = list(raw_dir.rglob('*.movie'))
    #%%
    cleaned_files_d = {x.stem : x for x in cleaned_files}
    raw_files_d = {x.stem : x for x in raw_files}
    
    #%%
    cols2smooth = ['roi_size', 'coord_x', 'coord_y']
    cols2copy = ['worm_index_joined', 'frame_number', 'worm_index_manual', 'threshold', 'worm_label', 'skeleton_id']
    
    posible_transitions = []
    for bn, cleaned_file in tqdm.tqdm(cleaned_files_d.items()):
        raw_file = raw_files_d[bn]
        
        trajectories_data = filter_trajectories(cleaned_file)
        
        
        df_smoothed = []
        for v_id, vescicle_data in trajectories_data.groupby('worm_index_joined'):
            track_size = vescicle_data['frame_number'].max() - vescicle_data['frame_number'].min() 
            if track_size < min_size:
                continue
                
            smoothed_data = {col:medfilt(vescicle_data[col].values, median_filter_size) for col in cols2smooth}
            for col in cols2copy:
                smoothed_data[col] = vescicle_data[col]
            smoothed_data = pd.DataFrame(smoothed_data)
            df_smoothed.append(smoothed_data)
        
            
            tt_trans = smoothed_data[smoothed_data['worm_label'] == 1.0]
            if len(tt_trans) > 0:
                tt_trans = tt_trans.iloc[0]['frame_number']
                if tt_trans > min_size:
                    posible_transitions += [tt_trans - 15, tt_trans, tt_trans + 15]
        
        if not df_smoothed:
            continue
        
        
        df_smoothed = pd.concat(df_smoothed, ignore_index=True)
        frames_dict = df_smoothed.groupby('frame_number').groups
        
        raw_reader = MovieReader(str(raw_file))
        fid = tables.File(str(cleaned_file), 'r')
        
        imgs_cleaned = fid.get_node('/mask')
        tot, img_h, img_w = imgs_cleaned.shape
        
        frames2read = [0, tot//4, tot//2, tot*3//4, tot*9//10, tot-1] + posible_transitions
        frames2read = [x for x in set(map(int, frames2read)) if x < tot-1]
        frames2read = [x for x in frames2read if x in frames_dict]
        
        save_dir = Path(str(cleaned_file.parent).replace(str(cleaned_dir), str(save_root_dir)))
        save_dir = save_dir / bn
        for ss in ['raw', 'cleaned', 'norm']:
            (save_dir / ss).mkdir(parents=True, exist_ok=True)
        
        
        for curr_frame in tqdm.tqdm(frames2read, desc = bn):
            img_cleaned = imgs_cleaned[curr_frame]
            _, img_raw = raw_reader[curr_frame]
            
            frame_data = df_smoothed.loc[frames_dict[curr_frame]]
            
            for _, row in frame_data.iterrows():
                v_id = int(row['worm_index_joined'])
                rr = int(math.ceil(row['roi_size']/2 + 5))
                cx = int(round(row['coord_x']))
                cy = int(round(row['coord_y']))
                
                roi_cleaned = img_cleaned[cy-rr:cy+rr, cx-rr:cx+rr]
                roi_raw = img_raw[cy-rr:cy+rr, cx-rr:cx+rr]
                roi_norm = norm_and_smooth(roi_cleaned)
                
                roi_bn = 'V{}_T{}.png'.format(v_id, curr_frame)
                cv2.imwrite(str(save_dir / 'raw' / roi_bn), roi_raw)
                cv2.imwrite(str(save_dir / 'cleaned' / roi_bn), roi_cleaned)
                cv2.imwrite(str(save_dir / 'norm' / roi_bn), roi_norm)
            
        
        fid.close()
        