#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:15:17 2018

@author: avelinojaver
"""
import tables
import cv2
import numpy as np
import pandas as pd

from collections import Counter
from skimage.filters import threshold_otsu

import tqdm


from extract_best_circles import extract_best_circles, norm_and_smooth
#%%
def get_largest_segment(val_b, max_gap = 10):
    
    val_shifted = np.concatenate((val_b[1:], [val_b[0]]))
    turn_on, = np.where(val_shifted & ~val_b)
    turn_off, = np.where(~val_shifted & val_b)
    
    turn_on = turn_on.tolist()
    turn_off = turn_off.tolist()
    
    
    if turn_off[0] < turn_on[0]:
        turn_off = turn_off[1:] + [len(val_b) + turn_off[0]]
    
    prev_on = turn_on[0]
    valid_pairs = []
    for ii in range(1, len(turn_on)):
        #get gap between the previous _off and the current _on
        #if this gap is too small ignore this gap and continue
        
        off_ = turn_off[ii-1]
        on_ = turn_on[ii]
        gap_ = (on_ - off_ )
        if gap_ > max_gap:
            valid_pairs.append((prev_on, off_))
            prev_on = on_
    
    off_ = turn_off[-1]
    on_ = (turn_on[0] + len(val_b))
    gap_ = (on_ - off_ )
    if gap_ <= max_gap:
        if len(valid_pairs) > 0:
            p1, p2 = valid_pairs[0]
            valid_pairs[0] = (prev_on, p2)
        else:
            #this is the conditions where after filling the gaps all the segment is valid
            valid_pairs.append((0, len(val_b)-1))
    else:
        valid_pairs.append((prev_on, off_))
    
    def _seg_size(x):
        _on, _off = x
        if _on > _off:
            _on += len(val_b)
        return _off - _on
    
    return max(valid_pairs, key=_seg_size)
#%%
def get_interfaces(img_n, mask_r, _debug = False):
    th = threshold_otsu(img_n[mask_r])
    mask = img_n > th
    mask[~mask_r] = False
    
    img_g = cv2.GaussianBlur(img_n, (11,11), 0 )
    img_g = cv2.Laplacian(img_g.astype(np.float64), ddepth=2, ksize=5)
    th = threshold_otsu(img_g)
    valid_mask = img_g > th
    valid_mask[~mask_r] = False
    valid_mask = valid_mask.astype(np.uint8)
    
    kernel = np.ones((3,3))
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
    valid_mask = cv2.dilate(valid_mask, kernel, iterations=5)
    
    im2, contours_ori, hierarchy  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy.squeeze(axis=0)
    
    #hierarchy -> [Next, Previous, First_Child, Parent]
    #let's do it simple. We just keep the largest contour and any children it might have (holes)
    assert hierarchy[0][1] == -1
    
    contours= []
    next_ = 0 
    while next_ != -1:
        curr_cnt = contours_ori[next_]
        contours.append(curr_cnt)
        
        #now let's add the primary holes of each major contour
        next_hole_ = hierarchy[next_][2] #first child
        while next_hole_ != -1:
            hole_cnt = contours_ori[next_hole_]
            contours.append(hole_cnt)
            next_hole_ = hierarchy[next_hole_][0]
        
        next_ = hierarchy[next_][0]
    
    
    contours_valid = []
    for cc in contours:
        
        cc = cc.squeeze(1)
        val_b = valid_mask[cc[..., 1], cc[..., 0]] > 0
        
        
        if np.all(val_b):
            contours_valid.append(cc)
        elif np.sum(val_b) >= 5:
            seg_pair = get_largest_segment(val_b)
            
            if seg_pair[1] > seg_pair[0]:
                if seg_pair[1] < len(val_b):
                    valid_c = cc[seg_pair[0]:seg_pair[1]+1]
                else:
                    fin = seg_pair[1]-len(val_b) + 1
                    c1 = cc[seg_pair[0]:]
                    c2 = cc[:fin+1]
                    valid_c = np.concatenate((c1,c2))
            else:
                c1 = cc[seg_pair[0]:]
                c2 = cc[:seg_pair[1]+1]
                valid_c = np.concatenate((c1,c2))
                
            contours_valid.append(valid_c)
    
    
    
    if _debug:
        import matplotlib.pylab as plt
        fig, axs = plt.subplots(1,3, figsize = (15, 5))
        axs[0].imshow(img_n)
        axs[1].imshow(mask)
        axs[2].imshow(valid_mask)
        
        for cc in contours_valid:
            axs[0].plot(cc[..., 0], cc[..., 1], 'r', lw=2)
        
        for cc in contours:
            axs[2].plot(cc[..., 0], cc[..., 1], 'r', lw=2)
    
    return contours_valid

#%%    

def get_intersections(current_contours,prev_contours, img_shape,  max_disp = 10, min_outer_frac = 1/16):
    rr = max_disp*2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rr, rr)) 
    
    overlaps = []
    for ii_curr, cc_curr in enumerate(current_contours):
        img_cc = np.zeros(img_shape, np.uint8)
        img_cc[cc_curr[..., 1], cc_curr[..., 0]] = 1
        
        img_cc = cv2.dilate(img_cc, kernel, iterations = 1)
        
        for ii_prev, cc_prev in enumerate(prev_contours):
            ovr = np.mean(img_cc[cc_prev[..., 1], cc_prev[..., 0]])
            overlaps.append((ii_curr, ii_prev, ovr))
        
    return overlaps
#%%
def track_interface(roi_file, 
                    min_seg_size = 25,
                    min_outer_frac = 1/16,
                    min_hole_frac = 1/32,
                    min_overlap = 0.5):
    #%%
    with tables.File(str(roi_file), 'r') as fid:
        imgs = fid.get_node('/mask')[:]
        tot, img_w, img_h = imgs.shape
    
    df_circle = extract_best_circles(roi_file)
    ini_frame = df_circle.index.min()
    
    #%%
    rx, ry = np.meshgrid(np.arange(img_w), np.arange(img_h))
    def _get_circle_mask(row):
        return np.sqrt((rx-row['cx'])**2 + (ry-row['cy'])**2) < row['radii']
    
    #get all possible interfaces
    interfaces_cnt = []
    for _, row in tqdm.tqdm(df_circle.iterrows(), total=len(df_circle)):
        if row['is_mixed'] !=  1:
            frame_number = int(row['frame_number']) - ini_frame
            mask_r = _get_circle_mask(row)
            
            img = imgs[frame_number]
            img_n = norm_and_smooth(img) 
            
            contours_valid = get_interfaces(img_n, mask_r)
            contours_valid = [cc for cc in contours_valid if len(cc) > min_seg_size]
            
            interfaces_cnt.append((frame_number, contours_valid))
        else:
            cc = None
    
    #%%
    prev_cnts = None
    index_prev = None
    
    all_contours = []
    traj_dat = []
    
    curr_cnt_ind = -1
    current_traj_id = 0
    for frame_number, frame_cnts in interfaces_cnt:
        index_current = list(range(current_traj_id+1, current_traj_id+len(frame_cnts)+1))
        current_traj_id += len(index_current)
        
        if index_prev is not None:
            #link data using overlap between blobs
            prev_intersect_ii = get_intersections(frame_cnts, prev_cnts, img_shape=(img_w, img_h))
            cur_intersect_ii = get_intersections(prev_cnts, frame_cnts, img_shape=(img_w, img_h))
            
            #only keep overlaping more than 0.5
            prev_intersect_f = [x[:2] for x in prev_intersect_ii if x[-1]>min_overlap]
            cur_intersect_f = [x[:2][::-1] for x in cur_intersect_ii if x[-1]>min_overlap]
            
            
            splitted_ind = [d[0] for d in Counter([x[1] for x in prev_intersect_f]).items() if d[1] > 1]
            merged_ind = [d[0] for d in Counter([x[0] for x in cur_intersect_f]).items() if d[1] > 1]
            
            
            #edges_splits = [(index_prev[x[1]], index_current[x[0]]) for x in prev_intersect_f if x[1] in splitted_ind]
            #edges_merges = [(index_prev[x[1]], index_current[x[0]]) for x in cur_intersect_f if x[0] in merged_ind]
                    
            all_intesect = set(cur_intersect_f) | set(prev_intersect_f)
                    
            simple_pairs = [x for x in all_intesect if not ((x[0] in merged_ind) or (x[1] in splitted_ind))]
            
            
            for i1, i2 in simple_pairs:
                index_current[i1] = index_prev[i2]
        
        #add to table
        for ind, cnt in zip(index_current, frame_cnts):
            curr_cnt_ind += 1
            
            cc = np.concatenate((np.full((cnt.shape[0], 1), curr_cnt_ind), cnt), axis=1)
            all_contours.append(cc)
            
            row_data = (ind, frame_number, curr_cnt_ind)
            traj_dat.append(row_data)
        
        
        
        prev_cnts = frame_cnts
        index_prev = index_current
     
        
    if len(all_contours) == 0:
        return
    
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    contours_data = pd.DataFrame(traj_dat, columns = ['interface_id', 'frame_number', 'contour_id'])
    contours_coordinates = pd.DataFrame(np.concatenate(all_contours), columns=['contour_id', 'X', 'Y'])
    #%%
    with tables.File(str(roi_file), 'r+') as fid:
        if '/contours_data' in fid:
            fid.remove_node('/', 'contours_data')
            
        if '/contours_coordinates' in fid:
            fid.remove_node('/', 'contours_coordinates')
        
        fid.create_table(
                '/',
                'contours_data',
                obj=contours_data.to_records(index=False),
                filters=TABLE_FILTERS)
        
        fid.create_table(
                '/',
                'contours_coordinates',
                obj=contours_coordinates.to_records(index=False),
                filters=TABLE_FILTERS)
        
        
    
#%%
if __name__ == '__main__':
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/raw_rois/ramp100-22Sep2016_17-49-11/roi_11.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/raw_rois/ramp40-29Oct2015_18-00-24/roi_3.hdf5'
    
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100-22Sep2016_17-49-11/roi_11.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/script_ramp-08Dec2015_16-45-56/roi_14.hdf5'
    roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp40.29Oct2015_18.00.24/roi_3.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100.29Oct2015_17.54.52/roi_1.hdf5'
    
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100-22Sep2016_17-49-11/roi_11.hdf5'
    roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100.22Sep2016_17.49.11/roi_24.hdf5'
    
    #%%
  
    track_interface(roi_file)
    
    