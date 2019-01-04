#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:44:22 2018

@author: avelinojaver
"""
import tables

fname = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/cleaned_rois/script_ramp.08Dec2015_16.45.56/roi_16.hdf5'


#%%
with tables.File(fname, 'r') as fid:
    imgs = fid.get_node('/mask')[:]
    #fid.get_node(fname)
#%%

med = np.median(imgs, axis= (1,2))
#%%
mad = np.median(np.abs(med[:, None, None]-imgs), axis=(1,2))
#%%

plt.figure()
plt.plot(med)

plt.figure()
plt.plot(mad)

#%%

#%%
fig, axs = plt.subplots(1,2, sharex=True, sharey=True)

th = med[-1] - mad[-1]

img = imgs[-1]
axs[0].imshow(img)
axs[1].imshow(img>th)
