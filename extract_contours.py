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
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter    
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.interpolate import RectBivariateSpline
#%%
def get_img_contours(img, _is_debug=False, img_o = None):
        
        img_b = cv2.medianBlur(img, 3)
        
        th_r = cv2.adaptiveThreshold(img_b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,151, 10)
        
        #this is scikit, those functions are slower but they are more convenient that using opencv
        th_r = remove_small_objects(th_r>0, min_size=100, connectivity=2)
        th_r = remove_small_holes(th_r, min_size=100, connectivity=2)
        
        thresh = th_r.astype(np.uint8)*255
        
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 1)
        
        im2, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #sort the found contours by area
        contours = sorted(contours, key=cv2.contourArea)[::-1]
        
            
        if _is_debug:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(thresh)
            
            plt.subplot(1,2,2)
            if img_o is not None:
                plt.imshow(img_o, interpolation='none', cmap='gray')
            for cc in contours[:2]:
                cc = np.squeeze(cc)
                plt.plot(cc[:, 0], cc[:, 1])
            plt.axis('equal')
            plt.axis('off')
            
        return contours
    
#tt = 200; get_img_contours(imgs_n[tt], True, imgs[tt])
#%%
def smooth_cnts(cnt, dist_th = 20):
        xx = cnt[:,0]
        yy = cnt[:,1]
          
        
        #check if there is not too much disntace between the points
        rr = np.hstack((0, np.sqrt(np.diff(xx)**2 + np.diff(yy)**2)))
        
        breaks_d, = np.where(rr>dist_th)
        #this is probably a fragmented contour, only keep the largest part
        if breaks_d.size > 0:
            dat = np.split(np.array((xx, yy)).T, breaks_d, axis=0)
            dat = max(dat, key = lambda x : x.size)
            xx, yy = dat.T
        
        
        #smooth
        xx_s = savgol_filter(xx, 25, 3)
        yy_s = savgol_filter(yy, 25, 3)
        
        rr_s = np.hstack((0, np.sqrt(np.diff(xx_s)**2 + np.diff(yy_s)**2)))
        rr_s = np.cumsum(rr_s)
        
        r_new = np.arange(0, rr_s[-1], 3) 
        
        xx_new = interp1d(rr_s, xx_s)(r_new)
        yy_new = interp1d(rr_s, yy_s)(r_new)
        
        return np.array((xx_new, yy_new)).T
#%%
def get_intensty(img_o, cc):
    mask = np.zeros(img_o.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cc[:, None, :]], -1, 1, -1)
    mask = mask==1
    
    in_pix = np.sum(mask)
    in_int = img_o[mask].sum()
    
    mask_n = ~mask
    out_pix = np.sum(mask_n)
    out_int = img_o[mask_n].sum()
    
    d1 = np.median(out_int/out_pix)
    d2 = np.median(in_int/out_int)
    if d1 > d2:
        in_pix, in_int, out_pix, out_int = out_pix, out_int, in_pix, in_int
    
    
    dat = (in_pix, in_int, out_pix, out_int)
    return dat
#%%
def get_int_profile(img, contour, half_width, width_resampling, _is_debug=False):
    
    dX = contour[1:, 0] - contour[:-1, 0]
    dY = contour[1:, 1] - contour[:-1, 1]
    cnt_angles = np.arctan2(dY, dX)
    
    #%get the perpendicular angles to define line scans (orientation doesn't
    #%matter here so subtracting pi/2 should always work)
    perp_angles = cnt_angles - np.pi / 2

    #%for each skeleton point get the coordinates for two line scans: one in the
    #%positive direction along perpAngles and one in the negative direction (use
    #%two that both start on skeleton so that the intensities are the same in
    #%the line scan)
    r_ind = np.linspace(-half_width, half_width, width_resampling)
    
    
    # create the grid of points to be interpolated (make use of numpy implicit
    # broadcasting Nx1 + 1xM = NxM)
    grid_y = contour[1:, 1] + r_ind[:, np.newaxis] * np.sin(perp_angles)
    grid_x = contour[1:, 0] + r_ind[:, np.newaxis] * np.cos(perp_angles)
    
    # interpolated the intensity map
    f = RectBivariateSpline(np.arange(img.shape[0]), np.arange(img.shape[1]), img)
    int_profile = f.ev(grid_y, grid_x)
    
    if _is_debug:
        plt.figure()
        plt.subplot(2,1,2)
        plt.imshow(int_profile)
        plt.axis('off')
        
        plt.subplot(2,2, 1)
        plt.plot(np.median(int_profile, axis=1))
        
        plt.subplot(2,2, 2)
        plt.imshow(img_o)
        plt.plot(grid_x, grid_y)
        plt.plot(contour[1:, 1], contour[1:, 0])
        plt.axis('off')
        
        
    
    return int_profile, cnt_angles, grid_x, grid_y

#lw=20; get_int_profile(img_o, cc, w_l//2, w_l*2, True)
#%%
if __name__ == '__main__':
    is_invert = False
    dist_th = 20
    
    #fname = '/Users/ajaver/OneDrive - Imperial College London/lucia/ramp100.29Oct2015_17.54.52.hdf5'
    #d_ranges = ((800,1075), (190,400), (210,360)); dist_th = np.inf; last_trusty = 200;
    #d_ranges = ((800,1075), (200,400), (185,250))
    
    fname = '/Users/ajaver/OneDrive - Imperial College London/lucia/ramp40.29Oct2015_18.00.24.hdf5'
    d_ranges = ((1325,1900), (200,395), (250,360)); last_trusty = 450
    #d_ranges = ((1125,1700), (200,390), (150,270)); is_invert = True
    #d_ranges = ((0,1900), (175,425), (150,360))
    
    #read data
    d_ranges = tuple([slice(*x) for x in d_ranges])
    with tables.File(fname, 'r') as fid:
        print(fid.get_node('/data'))
        imgs = fid.get_node('/data')[d_ranges]
    plt.imshow(imgs[0])
    #%% normalize intensity data
    bot, top = np.nanpercentile(imgs, [5, 95])
    imgs_n = ((imgs-bot)/(top-bot)*255).astype(np.uint8)
    if is_invert:
        imgs_n = 255-imgs_n
    #%% calculate all the contours
    all_cnts = [get_img_contours(x) for x in imgs_n]
    largest_cnt = [np.squeeze(x[0]) for x in all_cnts]
    #%% get intensity out and in the contour
    mask_ints = [get_intensty(*x) for x in zip(imgs, largest_cnt)]
    mask_ints = np.array(mask_ints)
    
    #%% filter and smooth contours
    d_lims_x = (1, imgs.shape[1]-1)
    d_lims_y = (1, imgs.shape[2]-1)
    contours_fixed = []
    for cc in largest_cnt: 
        good  = (cc[:,1]>d_lims_x[0]) & (cc[:,1]<d_lims_x[1])
        good  &= (cc[:,0]>d_lims_y[0]) & (cc[:,0]<d_lims_y[1])
        cc = cc[good]
    
        cc_s = smooth_cnts(cc, dist_th)
        contours_fixed.append(cc_s) 
    
    #%%
    light_pix, light_int, dark_pix, dark_int = mask_ints.T
    
    tot_int = light_int + dark_int
    tot_pix = light_pix + dark_pix
    
    plt.figure(figsize=(5, 5))
    
    ax1 = plt.subplot()#(1,2,1)
    plt.plot(light_int/light_pix)
    plt.plot(dark_int/dark_pix)
    ax2 = ax1.twinx()
    L = np.array([x.shape[0] for x in contours_fixed])
    #plt.figure()
    ax2.plot(L, color='gray')
    
    plt.plot((last_trusty, last_trusty), ax2.get_ylim(), ':r', lw=3)
    
#    plt.subplot(1,2,2)
#    plt.plot(light_int/tot_pix)
#    plt.plot(dark_int/tot_pix)
#    plt.plot(tot_int/tot_pix)
    
    
    
    #%%
    interface_profiles = []
    w_l = 10    
    for img_o, cc in zip(imgs, contours_fixed):
        int_profile_img, cnt_angle, _, _ = get_int_profile(img_o, cc, w_l//2, w_l*2)
        int_p = np.median(int_profile_img, axis=1)
        interface_profiles.append(int_p)
    interface_profiles = np.array(interface_profiles)
    
    plt.figure()
    plt.subplot(2,1,1)
    
    
    plt.plot(interface_profiles[:last_trusty].T)
    
    plt.subplot(2,1,2)
    plt.plot(interface_profiles[:last_trusty, 5])
    plt.plot(interface_profiles[:last_trusty, 8])
    plt.plot(interface_profiles[:last_trusty, 11])
    
            
    #%%
    tt = last_trusty
    cc = contours_fixed[tt]
    #cc = largest_cnt[tt]
    img_o = imgs[tt]
    
    straighten_worm, cnt_angles, grid_x, grid_y = get_int_profile(img_o, cc, w_l//2, w_l*2)
    plt.figure()
    plt.subplot(2,1,2)
    plt.imshow(straighten_worm)
    plt.axis('off')
    
    plt.subplot(2,2, 1)
    plt.plot(np.median(straighten_worm, axis=1))
    
    plt.subplot(2,2, 2)
    plt.imshow(img_o)
    plt.plot(grid_x, grid_y)
    plt.axis('off')
    
    
    
    
    plt.figure()
    
    plt.imshow(img_o, interpolation='none', cmap='gray')
    plt.plot(cc[:, 0], cc[:, 1], 'r', lw=2)
    
    plt.xlim(*d_lims_y)
    plt.ylim(*d_lims_x)
    plt.axis('off')
    #%%
    if False:
        for cc in contours_fixed[0::50]:
            ff = np.fft.fft(cc[:,0] + cc[:,1]*1j)
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(np.log10(np.abs(ff)))
            #plt.plot(np.abs(ff))
            plt.xlim(0, 100)
            
            plt.subplot(2,1,2)
            plt.plot(cc[:,1], cc[:,0])
            
            #plt.suptitle(tt) 
    
    #%%
    
    
    #%%
    
    dat = []
    #plt.figure()
    for tt in range(len(contours_fixed)):#range(350, 370):#range(0, len(largest_cnt), 50): 
        cc = contours_fixed[tt]
        ff = np.fft.fft(cc[:,0] + cc[:,1]*1j)
        ff2 = ff[:ff.size//2]
        
        dat.append(np.abs(ff2))
    tt = min([x.size for x in dat])
    dat_f = np.vstack([x[:tt] for x in dat])
    #%%
    #dat_n = dat_f/dat_f[0, :]/L
    #dat_n = dat_f/dat_f[0, :]
    dat_n = dat_f/dat_f[0, :]/L[:, None]
    plt.figure()
    
    dd = [0, 10, 25, 50, 75]
    #dd = [0,  50, 100, 120]
    for ii in dd:
        plt.plot(savgol_filter(dat_n[:, ii], 5, 3))
    
    #plt.plot(dat_n[:, 50])
    
    
    #plt.plot(dat_n[:, 100])
    #plt.xlim((300, 450))
    #%%
    if True:
        plt.figure(figsize=(10, 5))
        #dd = [340, 350, 360, 370, 380, 390, 400]
        #dd = [140, 150, 160, 170]
        dd = [170, 180, 190, 200, 230]
        #dd = [53, 100, 110]
        for ii, tt in enumerate(dd):
            plt.subplot(1,len(dd),ii + 1)
            plt.imshow(imgs[tt])
            
            cc = contours_fixed[tt]
            plt.plot(cc[:, 0], cc[:, 1], 'r', lw=2)
            
            plt.title(tt)
    
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
        l, = plt.plot([], [], 'r', lw=2)
        
        img_o = imgs[0]
        canvas = plt.imshow(img_o, interpolation='none', cmap='gray')
        
        plt.xlim(*d_lims_y)
        plt.ylim(*d_lims_x)
        
        x0, y0 = 0, 0
        
        tot = len(largest_cnt)
        save_name = fname.replace('.hdf5', '_patch.mp4')
        with writer.saving(fig, save_name, tot):
            for tt in range(last_trusty):
                
                cc = contours_fixed[tt]
                img_o = imgs[tt]
                
                canvas.set_array(img_o)
                
                l.set_data(cc[:, 0], cc[:, 1])
                writer.grab_frame()