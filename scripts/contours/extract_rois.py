#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:56:47 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))
from read_movies.moviereader import MovieReader

import pandas as pd
import tables
from scipy.signal import medfilt
import tqdm
import math

TABLE_FILTERS = tables.Filters(
        complevel = 9,
        complib = 'zlib',
        shuffle = True,
        fletcher32 = True)
    

def filter_trajectories(movie_name, 
                        min_good_fraction = .6, 
                        min_vesicle_radius = 100,
                        ignore_mixed = True):
    
    #read data
    with pd.HDFStore(str(movie_name), 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    if len(trajectories_data) == 0:
        return
    
    with tables.File(str(movie_name), 'r') as fid:
        tot_frames, img_h, img_w = fid.get_node('/mask').shape
    
    #only keep trajectories that where tracked for at least `min_good_fraction` of the video
    tot_frames = trajectories_data['frame_number'].max() + 1
    traj_length = trajectories_data['worm_index_joined'].value_counts()
    
    min_frames = math.floor(min_good_fraction*tot_frames)
    valid_index_1 = traj_length.index[traj_length >= min_frames]
    
    #only keep vesicles that are at least `min_vesicle_radius`
    med_R = trajectories_data.groupby('worm_index_joined').agg({'roi_size':'median'})['roi_size']
    valid_index_2 = med_R.index[med_R>min_vesicle_radius]
    
    valid_index = set(valid_index_1) & set(valid_index_2)
    trajectories_data = trajectories_data[trajectories_data['worm_index_joined'].isin(valid_index)]
    
    #ignore rois that touch the border of the video
    rr = trajectories_data['roi_size']/2
    bad_x_l = (trajectories_data['coord_x'] - rr)<0
    bad_y_l = (trajectories_data['coord_y'] - rr)<0
    bad_x_r = (trajectories_data['coord_x'] + rr) > img_w
    bad_y_r = (trajectories_data['coord_y'] + rr) > img_h
    
    trajectories_data['in_image'] = ~(bad_x_l | bad_y_l | bad_x_r | bad_y_r)
    
    valid_index = trajectories_data.groupby('worm_index_joined').agg({'in_image':'all'})['in_image']
    try:
        valid_index = valid_index.index[valid_index]
    except:
        import pdb
        pdb.set_trace()
    
    
    if ignore_mixed:
        is_all_mixed = trajectories_data.groupby('worm_index_joined').agg({'worm_label':'max'})
        is_all_mixed = is_all_mixed['worm_label'] == 1
        valid_index = set(is_all_mixed.index[~is_all_mixed]) & set(valid_index)
        
    
    filtered_trajectories = trajectories_data[trajectories_data['worm_index_joined'].isin(valid_index)]
    #%%
    return filtered_trajectories


def extract_rois(movie_name, 
                 save_dir, 
                 raw_movie = None,  
                 median_filter_size = 25,
                 **argkws):
    
    min_size =  median_filter_size*2
    
    movie_name = Path(movie_name)
    save_dir = Path(save_dir)
    if raw_movie is not None:
        raw_movie = Path(raw_movie)
    
    trajectories_data = filter_trajectories(movie_name, **argkws)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cols2smooth = ['roi_size', 'coord_x', 'coord_y']
    #I add this fields just to be compatible with MWTrackerViewer
    cols2copy = ['worm_index_joined', 'frame_number', 'worm_index_manual', 'threshold', 'worm_label', 'skeleton_id']
    
    vesicles_rois_fid = {}
    
    df_smoothed = []
    for v_id, vescicle_data in trajectories_data.groupby('worm_index_joined'):
        if len(vescicle_data) < min_size:
            continue
        
        smoothed_data = {col:medfilt(vescicle_data[col].values, median_filter_size) for col in cols2smooth}
        for col in cols2copy:
            smoothed_data[col] = vescicle_data[col]
        smoothed_data = pd.DataFrame(smoothed_data)
            
        #use the maximum size found after filtering
        roi_size = math.ceil(smoothed_data['roi_size'].max()/2)*2
        
        
        ini_frame = smoothed_data['frame_number'].min()
        tot_frames =  smoothed_data['frame_number'].max() - ini_frame + 1
        
        #initialize the roi files
        v_file = save_dir / 'roi_{}.hdf5'.format(v_id)
        roi_fid = tables.File(str(v_file), 'w')
        
        roi_fid.create_carray(
                            '/',
                            'mask',
                            tables.UInt16Atom(),
                            shape = (tot_frames, roi_size, roi_size),
                            chunkshape = (1, roi_size, roi_size),
                            filters = TABLE_FILTERS)
        
        tab2save = smoothed_data.copy()
        cc = roi_size/2.
        tab2save['coord_x'] = cc
        tab2save['coord_y'] = cc
        
        roi_fid.create_table(
                            '/',
                            'trajectories_data',
                            obj = tab2save.to_records(index=False),
                            filters = TABLE_FILTERS)
        
        vesicles_rois_fid[v_id] = (ini_frame, roi_fid)
        df_smoothed.append(smoothed_data)
    
    if len(df_smoothed) == 0:
        return
    
    
    df_smoothed = pd.concat(df_smoothed, ignore_index=True)
    
    
    if raw_movie is not None:
        reader = MovieReader(raw_movie)
    else:
        fid = tables.File(str(movie_name), 'r')
        reader = fid.get_node('/mask')
    
    
    for curr_frame, frame_data in tqdm.tqdm(df_smoothed.groupby('frame_number')):
        img = reader[curr_frame]
        if isinstance(img, tuple):
            img = img[1]
        
        
        for _, row in frame_data.iterrows():
            v_id = int(row['worm_index_joined'])
            
            ini_frame, roi_fid = vesicles_rois_fid[v_id]
            
            rois = roi_fid.get_node('/mask')
            _, ves_h, ves_w = rois.shape
            
            rr = int(math.ceil(ves_h/2))
            cx = int(round(row['coord_x']))
            cy = int(round(row['coord_y']))
            
            rois[curr_frame - ini_frame] = img[cy-rr:cy+rr, cx-rr:cx+rr]

    if raw_movie is not None:
        fid.close()
                    
    for v_id, (_, fid) in vesicles_rois_fid.items():
        fid.close()


#%%
if __name__ == '__main__':
    
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/ramp100.22Sep2016_17.49.11.hdf5'
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/script_ramp.08Dec2015_16.45.56.hdf5'
    movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/script_ramp.08Dec2015_17.09.35.hdf5'
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/ramp100.29Oct2015_17.54.52.hdf5'
    #movie_name = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned/ramp40.29Oct2015_18.00.24.hdf5'
    
    #_read_raw = False
    save_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/')
    
    #_read_raw = True
    #save_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/vesicle/raw_rois/')
    #extract_rois(movie_name,  save_root_dir, min_good_fraction=0.5)
    
    trajectories_data = filter_trajectories(movie_name, min_good_fraction = 0.5, min_vesicle_radius = 100)
    
    