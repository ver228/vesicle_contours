#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:46:25 2018

@author: avelinojaver
"""
import cv2
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import disk
from skimage.draw import circle_perimeter

import pandas as pd
import numpy as np

import tqdm

#%%
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel


def get_best_circles(mask, min_factor = 0.5,  max_factor=1.2, max_center_offset = 0.2, resize_factor = 8):
    '''
    Get the best the best fit to a circle using the hough transform.
    '''
    
    #resize image to increase speed. I don't want 
    
    min_size = min(mask.shape)
    resize_factor = min_size/max(64, min_size/resize_factor)
    dsize = tuple(int(x/resize_factor) for x in mask.shape[::-1])
    
    mask_s = cv2.dilate(mask, disk(resize_factor/2))
    mask_s = cv2.resize(mask_s,dsize)>0
    
    r_min = min(mask_s.shape)*min_factor
    r_max = max(mask_s.shape)*max_factor
    
    
    
    
    #use the first peak of the circle hough transform to initialize the food shape
    hough_radii = np.arange(r_min, r_max, 2)
    hough_res = hough_circle(mask_s, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=9)
    
    
    cx, cy, radii = [np.round(x*resize_factor).astype(np.int) for x in (cx, cy, radii)]
    
    img_center = min_size//2
    offset_ = min_size*max_center_offset
    
    offset_l, offset_r = img_center-offset_ , img_center+offset_
    h_r = []
    for hh in list(zip(accums, cx, cy, radii)):
        if hh[1] >=  offset_l and hh[1] <= offset_r and hh[2] >=  offset_l and hh[2] <= offset_r:
            h_r.append(hh)
    
    
    return h_r

def norm_and_smooth(img, q = 1, n_iteration = 8):
    bot, top = np.nanpercentile(img, [q, 100 - q])
    img_n = ((img-bot)/(top-bot)*255)
    img_n = np.clip(img_n, 0, 255).astype(np.uint8)
    
    for _ in range(8):
        img_n = cv2.medianBlur(img_n, 3)
    
    return img_n

def extract_best_circles(fname):
    main_circles = []
    with pd.HDFStore(str(fname), 'r') as fid:
        src = fid.get_node('/mask')
        roi_data = fid['/trajectories_data']
        
        frames_gg = roi_data.groupby('frame_number')
        ini_frame = roi_data['frame_number'].min()
        
        for frame_number, frame_data in tqdm.tqdm(frames_gg):
        
            row = frame_data.iloc[0]
            cx = row['coord_x']
            cy = row['coord_y']
            rr = row['roi_size']//2 + 5
            
            ini_x = max(cx-rr, 0)
            ini_y = max(cy-rr, 0)
            
            fin_x = min(src.shape[1], cx+rr+1)
            fin_y = min(src.shape[1], cy+rr+1)
            
            tt = frame_number - ini_frame
            img = src[tt, ini_x:fin_x, ini_y:fin_y]
            
            img_n = norm_and_smooth(img)
            
            
            bb = int(round(min(img_n.shape)/4))
            bb = bb if bb % 2 == 1 else bb + 1
            mask = cv2.adaptiveThreshold(img_n,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,bb, 0)
            mask_skel = skeletonize(mask)
            
            
            h_res = get_best_circles(mask_skel)
            if len(h_res) > 0:
                #_, cy0, cx0, cr0 =  min(h_res, key=_closest_center)
                _, cy0, cx0, cr0 = h_res[0]
                
                row = (tt, cy0 + ini_y, cx0 + ini_x, cr0, row['worm_label'])
            else:
                row = (tt, np.nan, np.nan, np.nan, row['worm_label'])
            
            main_circles.append(row)
    
    df = pd.DataFrame(main_circles, columns = ['frame_number', 'cx', 'cy', 'radii', 'is_mixed'])
    
    cols2smooth = ['cx', 'cy', 'radii']
    cols2copy = ['frame_number', 'is_mixed']
    
    df_circle = df[cols2smooth].rolling(25).median()
    for col in cols2copy:
        df_circle[col] = df[col]
        
    df_circle.fillna(method='ffill',  inplace=True)
    df_circle.fillna(method='bfill',  inplace=True)
    
    return df_circle