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

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.signal import medfilt

from skimage.draw import circle_perimeter

import tqdm


from extract_best_circles import  norm_and_smooth, get_best_circles, skeletonize, extract_best_circles




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

def _old_interface_mask(img_n, mask_r):
    #%%
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

def get_intersections(current_contours,prev_contours, img_shape,  max_disp = 2, min_outer_frac = 1/16):
    
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
def segment_interfaces(roi_file, min_seg_size):
    with tables.File(str(roi_file), 'r') as fid:
        imgs = fid.get_node('/mask')[:]
        tot, img_w, img_h = imgs.shape
    
    df_circle = extract_best_circles(roi_file)
    ini_frame = df_circle.index.min()
    
    
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
        
            
    return interfaces_cnt


    
    
    #%%
#%%

#%%
def _possible_frame_circles(img_n, top_acc_frac = 0.9):
    
    kernel = np.ones((3,3))
    bb = int(round(min(img_n.shape)/4))
    bb = bb if bb % 2 == 1 else bb + 1
    mask = cv2.adaptiveThreshold(img_n,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,bb, 0)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    _, contours_ori, hierarchy  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    mask_skel = skeletonize(mask)
    
    borders = mask_skel.copy()
    cv2.drawContours(borders, contours_ori, -1, 255, 1)
    h_res = get_best_circles(borders)
    
    if len(h_res) == 0:
        return
    
    top_acc = h_res[0][0]
    acc_th = top_acc*top_acc_frac
    
    h_res = [x for x in  h_res if x[0]> acc_th]
    
    
    return h_res

#%%
def _get_circle_masks(imgs_n):
    #%%
    MAX_N_CIRCLES = 2
    
    prev_circs = None
    
    def _dist(x1, y1, x2, y2):
        return (x1-x2)**2 + (y1-y2)**2
        
    all_circles = []
    for img_n in tqdm.tqdm(imgs_n):
        cur_circs = _possible_frame_circles(img_n)
        if cur_circs is None:
            if prev_circs is None:
                rr = img_n.shape[0]//2
                cur_circs = [(0, rr, rr, rr)]
            else:
                cur_circs = prev_circs
        
        if len(cur_circs) < MAX_N_CIRCLES:
            cur_circs = cur_circs + [cur_circs[0]]*(MAX_N_CIRCLES - len(cur_circs))
        cur_circs = cur_circs[:MAX_N_CIRCLES]
        
        cur_circs = np.array(cur_circs)
        
        
        if prev_circs is not None:
            cost = cdist(prev_circs[:, 1:3], cur_circs[:, 1:3])
            row_ind, col_ind = linear_sum_assignment(cost)
            cur_circs = cur_circs[col_ind]

            
        all_circles.append(cur_circs)
        prev_circs = cur_circs
        
    
    circs = np.array(all_circles)
    circs_smoothed = circs.copy()
    
    
    filt_size = max(11, len(imgs_n)//10)
    filt_size = filt_size if filt_size % 2 else filt_size + 1
    
    
    for cc in [1,2,3]:
        for jj in range(2):
            xx = circs[:, jj, cc]
            xx = np.pad(xx, (filt_size, filt_size), 'symmetric')
            xx = medfilt(xx, filt_size)
            xx = xx[filt_size:-filt_size]
            
            circs_smoothed[:, jj, cc] = xx
            
    
    return circs_smoothed
    
#%%
def _get_circle_mask(circs, img_size, radii_fractor = 0.9):
    mask_rs = []
    for row in circs:
        cx, cy, cr = map(int, row[1:])
        circ_cnt = np.stack([x[:, None] for x in circle_perimeter(cx, cy, int(cr*radii_fractor))], axis=2)
        dd = np.zeros(img_size, np.uint8)
        cv2.drawContours(dd, [circ_cnt], -1, 255, -1)
        mask_rs.append(dd)
    mask_union = np.any(np.array(mask_rs), axis=0)
    mask_intesect  = np.all(np.array(mask_rs), axis=0)
    
    return mask_union, mask_intesect
    
#%%
def _get_frame_interface(img_n, circs, min_seg_size):
    
    mask_union, mask_intesect = _get_circle_mask(circs, img_n.shape)
    
    sigma = img_n.shape[0]//2
    k = sigma*2 + 1
    
    img_s = img_n.astype(np.float32) - cv2.GaussianBlur(img_n, (k, k), sigma, sigma, borderType = cv2.BORDER_REFLECT_101)
    
    bot, top = np.min(img_s), np.max(img_s)
    img_s = (img_s - bot) / (top-bot)
    
    th = threshold_otsu(img_s[mask_intesect])
    valid_mask = (img_s > th) * mask_union
    
    im2, cnts, _  = cv2.findContours(valid_mask.astype(np.uint8), 
                                                     cv2.RETR_LIST, 
                                                     cv2.CHAIN_APPROX_NONE)
    
    
    im2, cnt_external, _  = cv2.findContours(mask_union.astype(np.uint8), 
                                                     cv2.RETR_LIST, 
                                                     cv2.CHAIN_APPROX_NONE)
    
    border = np.zeros(img_s.shape, dtype = np.uint8)
    cv2.drawContours(border, cnts, -1, 1)
    cv2.drawContours(border, cnt_external, -1, 0)
    
    im2, cnts, _  = cv2.findContours(border, 
                                         cv2.RETR_CCOMP, 
                                         cv2.CHAIN_APPROX_NONE)
    
    cnts = [x.squeeze(axis=1) for x in cnts if x.shape[0] > min_seg_size]
    
    
    # it seems that the findContours algorithm produces replicated values, this is a hack to try to fix it
    valid_cnts= []
    for cnt in cnts:
         cnt_pixels = border[cnt[...,1], cnt[..., 0]]
        
         is_valid = (cnt_pixels==1).any()
         if is_valid:
             valid_cnts.append(cnt)
             border[cnt[...,1], cnt[..., 0]] = 0
    
    
    cnt_external = cnt_external[0].squeeze(axis=1)
    
    return valid_cnts,  cnt_external
#%%

#def _segment_interfaces(roi_file, min_seg_size):
#    
#    with tables.File(str(roi_file), 'r') as fid:
#        imgs = fid.get_node('/mask')[:]
#    
#    def _filter_cnts(img):
#        cnts = _get_frame_interface(img)
#        return [x.squeeze(axis=1) for x in cnts if x.shape[0] > min_seg_size]
#    
#    
#    interfaces_cnt = [(frame, _filter_cnts(img)) for frame, img in enumerate(tqdm.tqdm(imgs))]
#    
#    return interfaces_cnt
    
#%%


    

#%%
def track_interface(roi_file, 
                    min_seg_size = 25,
                    min_overlap = 0.5):
    #%%
    with tables.File(str(roi_file), 'r') as fid:
        imgs = fid.get_node('/mask')[:]
        tot, img_w, img_h = imgs.shape
    
    imgs_n = [norm_and_smooth(x) for x in imgs]
    
    circs_smoothed = _get_circle_masks(imgs_n)
    #%%
    cnt_data = [_get_frame_interface(img_n, circs, min_seg_size) 
    for img_n, circs in tqdm.tqdm(zip(imgs_n, circs_smoothed), total=len(imgs_n))]
    #%%
    prev_cnts = None
    index_prev = None
    prev_center = None
    
    all_contours = []
    traj_dat = []
    
    curr_cnt_ind = -1
    
    external_cnt_id = 1# I am reserving the current_traj_id = 1 to the external contour
    current_traj_id = 1 
    
    for frame_number, (interface_cnt, external_cnt) in enumerate(cnt_data):
        
        cur_center = np.mean(external_cnt, axis=0)
       
        index_current = list(range(current_traj_id+1, current_traj_id+len(interface_cnt)+1))
        current_traj_id += len(index_current)
        
        if index_prev is not None:
            delta_center = (cur_center - prev_center).astype(np.int)
            prev_cnt_ss = [cnt + delta_center[None, :] for cnt in prev_cnts]
            prev_cnt_ss = [np.clip(cnt, 0,img_w-1) for cnt in prev_cnt_ss]
            
            #link data using overlap between blobs
            prev_intersect_ii = get_intersections(interface_cnt, prev_cnt_ss, img_shape=(img_w, img_h))
            cur_intersect_ii = get_intersections(prev_cnt_ss, interface_cnt, img_shape=(img_w, img_h))
            
            #only keep overlaping more than 0.5
            prev_intersect_f = [x[:2] for x in prev_intersect_ii if x[-1]>min_overlap]
            cur_intersect_f = [x[:2][::-1] for x in cur_intersect_ii if x[-1]>min_overlap]
            
            
            splitted_ind = [d[0] for d in Counter([x[1] for x in prev_intersect_f]).items() if d[1] > 1]
            merged_ind = [d[0] for d in Counter([x[0] for x in cur_intersect_f]).items() if d[1] > 1]
            
            
#            if frame_number == 14:
#                #%%
#                plt.figure()
#                for ii, cnt in enumerate(prev_cnts):
#                    plt.plot(cnt[...,0], cnt[...,1], 'r')
#                    plt.text(cnt[0,0], cnt[0,1], str(ii), color='r')
#                    
#                
#                for ii, cnt in enumerate(interface_cnt):
#                    plt.plot(cnt[...,0], cnt[...,1], 'b')
#                    plt.text(cnt[0,0], cnt[0,1], str(ii), color='b')
                
           
            
            #edges_splits = [(index_prev[x[1]], index_current[x[0]]) for x in prev_intersect_f if x[1] in splitted_ind]
            #edges_merges = [(index_prev[x[1]], index_current[x[0]]) for x in cur_intersect_f if x[0] in merged_ind]
                    
            all_intesect = set(cur_intersect_f) | set(prev_intersect_f)
                    
            simple_pairs = [x for x in all_intesect if not ((x[0] in merged_ind) or (x[1] in splitted_ind))]
            
            
            for i1, i2 in simple_pairs:
                index_current[i1] = index_prev[i2]
        
        #add to table
        for ind, cnt in zip(index_current, interface_cnt):
            curr_cnt_ind += 1
            
            cc = np.concatenate((np.full((cnt.shape[0], 1), curr_cnt_ind), cnt), axis=1)
            all_contours.append(cc)
            
            row_data = (ind, frame_number, curr_cnt_ind)
            traj_dat.append(row_data)
        
        #add external contour
        curr_cnt_ind += 1
        cc = np.concatenate((np.full((external_cnt.shape[0], 1), curr_cnt_ind), external_cnt), axis=1)
        all_contours.append(cc)
        
        row_data = (external_cnt_id, frame_number, curr_cnt_ind)
        traj_dat.append(row_data)
        
        
        prev_cnts = interface_cnt
        index_prev = index_current
        prev_center = cur_center
        
    if len(all_contours) == 0:
        return
    
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    contours_data = pd.DataFrame(traj_dat, columns = ['interface_id', 'frame_number', 'contour_id'])
    contours_coordinates = pd.DataFrame(np.concatenate(all_contours), columns=['contour_id', 'X', 'Y'])
    
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
    
    
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/script_ramp.08Dec2015_16.45.56/roi_4.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100.29Oct2015_17.54.52/roi_1.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp40.29Oct2015_18.00.24/roi_3.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/ramp100.22Sep2016_17.49.11/roi_24.hdf5'
    
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/roi_cleaned/08_12_15/roomT75_100_20/script_ramp.08Dec2015_15.31.49/roi_28.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/roi_cleaned/08_12_15/roomT75_100_20/script_ramp.08Dec2015_15.31.49/roi_1.hdf5'
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/roi_cleaned/08_12_15/roomT8_100_20/script_ramp.08Dec2015_18.45.08/roi_10.hdf5'
    
    #roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/roi_cleaned/08_12_15/roomT75_100_20bis/script_ramp.09Dec2015_17.44.33/roi_24.hdf5'
    roi_file = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/roi_cleaned/08_12_15/roomT8_100_20/script_ramp.08Dec2015_19.13.59/roi_46.hdf5'
    #%%
  
    track_interface(roi_file)
    
    