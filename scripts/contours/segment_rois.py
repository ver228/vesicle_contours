#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:03:14 2018

@author: avelinojaver
"""
import sys
from pathlib import Path
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import threshold_otsu
from skimage.morphology import disk

import cv2
import random
from skimage.draw import circle_perimeter
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


def get_best_circles(mask, min_factor = 0.5,  max_factor=1.2, max_center_offset = 0.3, resize_factor = 8):
    '''
    Get the best the best fit to a circle using the hough transform.
    '''
    
    #resize image to increase speed. I don't want 
    
    min_size = min(mask.shape)
    resize_factor = min_size/max(64, min_size/resize_factor)
    dsize = tuple(int(x/resize_factor) for x in mask.shape[::-1])
    
    mask_s = cv2.dilate(mask, disk(resize_factor/2))
    mask_s = cv2.resize(mask_s, dsize)>0
    
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
#%%
    
if __name__ == '__main__':
    src_dir = Path.home() / 'workspace/Vesicles/segmented/'
    roi_files = list(src_dir.rglob('norm/*.png'))
    #%%
    
    
    random.shuffle(roi_files)
    
    #%%
    kernel = np.ones((3,3))
    for ii, roi_file in enumerate(roi_files):
        
        img = cv2.imread(str(roi_file), -1)
        
        bb = int(round(min(img.shape)/4))
        bb = bb if bb % 2 == 1 else bb + 1
        mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,bb, 0)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        _, contours_ori, hierarchy  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        mask_skel = skeletonize(mask)
        
        borders = mask_skel.copy()
        cv2.drawContours(borders, contours_ori, -1, 255, 1)
        h_res = get_best_circles(borders)
        
        mask_rs = []
        top_acc = h_res[0][0]
        for acc, cx, cy, cr in h_res[0:2]:
            if acc > top_acc*0.9:
                circ_cnt = np.stack([x[:, None] for x in circle_perimeter(cy, cx, int(cr*0.9))], axis=2)
                dd = np.zeros_like(borders)
                cv2.drawContours(dd, [circ_cnt], -1, 255, -1)
                mask_rs.append(dd)
                
            
        mask_union = np.any(np.array(mask_rs), axis=0)
        mask_intesect  = np.all(np.array(mask_rs), axis=0)
        
        
        
        #bb = int(round(min(img.shape)/16))
        #bb = bb if bb % 2 == 1 else bb + 1
        #mask_l = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #            cv2.THRESH_BINARY,bb, -2) > 0
        
        
        sigma = img.shape[0]//2
        k = sigma*2 + 1
        
        img_s = img.astype(np.float32) - cv2.GaussianBlur(img, (k, k), sigma, sigma, borderType = cv2.BORDER_REFLECT_101)
        
        #mask_f = cv2.erode((mask_intesect).astype(np.uint8), kernel, iterations=5) > 0
        mask_f = mask_intesect > 0
        
        bot, top = np.min(img_s), np.max(img_s)
        img_s = (img_s - bot) / (top-bot)
        
        th = threshold_otsu(img_s[mask_f])#*1.1
        valid_mask = (img_s > th) * mask_union
                                       
        #th = threshold_otsu(img[(mask_intesect) & (mask_l==0)])*1.1
        #mask_l2 = img> th
        
        #valid_mask1 = (mask_l2) & (mask_union > 0)
        #valid_mask2 = (mask_l2) & (mask_intesect > 0)
        
        
        
        fig, axs = plt.subplots(1,3, figsize = (15, 5), sharex=True, sharey=True)
        axs[0].imshow(img)
        axs[1].imshow(img_s)
        axs[2].imshow(valid_mask)
        
        #axs[1].imshow(valid_mask1)
        #axs[2].imshow(valid_mask2)
        
        for (acc, cx, cy, cr) in h_res[0:2]:
            if acc > top_acc*0.9:
                cpx,cpy = circle_perimeter(cy, cx, int(cr*0.9))
                axs[0].plot(cpx,cpy, '.')
        
        
        
        
        
        
        
#        
#        fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
#        axs = axs.flatten()
#        axs[0].imshow(img)
#        axs[1].imshow(dd)
#        
#        th = threshold_otsu(dd)
#        axs[2].imshow(dd>th)
        
        #th2 = threshold_otsu(dd[dd<th])
        #axs[3].imshow(dd>th2)
        
        if ii > 20:
            break
        
    #%%
    
    
    #%%
    