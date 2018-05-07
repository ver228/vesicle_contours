#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:27:39 2018

@author: ajaver
"""
import tables
import matplotlib.pylab as plt
import cv2
import numpy as np
    
def get_img_contours(img, _is_debug=False, img_o = None):
        img_b = cv2.medianBlur(img, 3)
        
        th_r = cv2.adaptiveThreshold(img_b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,151, 10)
        
        #this is scikit, those functions are slower but they are more convenient that using opencv
        th_r = remove_small_objects(th_r>0, min_size=100, connectivity=2)
        th_r = remove_small_holes(th_r, min_size=100, connectivity=2)
        
        thresh = th_r.astype(np.uint8)*255
        im2, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #sort the found contours by area
        contours = sorted(contours, key=cv2.contourArea)[::-1]
        
            
        if _is_debug:
            
            plt.figure()
            if img_o is not None:
                plt.imshow(img_o, interpolation='none', cmap='gray')
            for cc in contours[-2:]:
                cc = np.squeeze(cc)
                plt.plot(cc[:, 0], cc[:, 1])
            plt.axis('equal')
            plt.axis('off')
            
        return contours
if __name__ == '__main__':
    #fname = '/Users/ajaver/OneDrive - Imperial College London/lucia/ramp100.29Oct2015_17.54.52.hdf5'
    #d_ranges = ((800,1075), (200,400), (170,360))
    
    fname = '/Users/ajaver/OneDrive - Imperial College London/lucia/ramp40.29Oct2015_18.00.24.hdf5'
    d_ranges = ((1325,1900), (175,425), (150,360))
    
    d_ranges = tuple([slice(*x) for x in d_ranges])
    with tables.File(fname, 'r') as fid:
        imgs = fid.get_node('/data')[d_ranges]#[780:1075, 200:400, 170:360]
    plt.imshow(imgs[-1])
    #%%
    bot, top = np.nanpercentile(imgs, [5, 95])
    #bot, top = np.nanpercentile(imgs, [0, 100])
    
    
    imgs_n = ((imgs-bot)/(top-bot)*255).astype(np.uint8)
    
    #%%
    from skimage.morphology import remove_small_objects, remove_small_holes
    
    all_cnts = [get_img_contours(x) for x in imgs_n]
    #for tt in range(0, imgs_n.shape[0], 20):
    #    cnts = get_img_contours(img, _is_debug=False, img_o = None)
    #    cnts.append(all_cnts)
    #%%
    largest_cnt = [np.squeeze(x[0]) for x in all_cnts]
    #%%
    d_lims_x = (25, 220)
    d_lims_y = (100, 210)
    
    tt = -100
    
    cc = largest_cnt[tt]
    img_o = imgs[tt].T
    
    good  = (cc[:,1]>d_lims_x[0]) & (cc[:,1]<d_lims_x[1])
    good  &= (cc[:,0]>100) 
    cc_r = cc[good]
    
    
    plt.figure()
    
    plt.imshow(img_o, interpolation='none', cmap='gray')
    plt.plot(cc_r[:, 1], cc_r[:, 0], 'r', lw=2)
    
    plt.xlim(*d_lims_x)
    plt.ylim(*d_lims_y)
    plt.axis('off')
    #plt.axis('equal')
    #%%
    if True:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation
        
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        
        fig = plt.figure()
        l, = plt.plot([], [], '-')
        
        canvas = plt.imshow(img_o, interpolation='none', cmap='gray')
        
        plt.xlim(*d_lims_x)
        plt.ylim(*d_lims_y)
        
        x0, y0 = 0, 0
        
        tot = len(largest_cnt)
        with writer.saving(fig, "writer_test.mp4", tot):
            for tt in range(tot):
                
                cc = largest_cnt[tt]
                img_o = imgs[tt].T
                
                good  = (cc[:,1]>25) & (cc[:,1]<225)
                good  &= (cc[:,0]>100) 
                cc_r = cc[good]
        
                canvas.set_array(img_o)
                
                l.set_data(cc_r[:, 1], cc_r[:, 0])
                writer.grab_frame()