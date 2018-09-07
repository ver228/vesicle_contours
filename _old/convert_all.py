#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:21:47 2018

@author: avelinojaver
"""
import tables
import os
import glob
import tqdm

from moviereader import MovieReader, IIDC_header_dtype

def movie_to_hdf5(fname, save_name = None, is_force = False):
    if save_name is None:
        save_name = fname.replace('.movie', '.hdf5')
    
    try:
         with tables.File(save_name, 'r') as fid:
             is_finished = fid.get_node('images')._v_attrs['is_finished']>0
             if is_finished and not is_force:
                 return
    except:
        pass
                 
        
    
    mreader = MovieReader(fname)
    
    
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    with tables.File(save_name, 'w') as fid_save:
        if mreader._data_depth == 8:
            atom = tables.UInt8Atom()
        else:
            atom = tables.UInt16Atom()
        
        images = fid_save.create_carray(
                        '/',
                        'images',
                        atom,
                        shape = (mreader.number_of_frames, mreader.height, mreader.width),
                        chunkshape = (1, mreader.height, mreader.width),
                        filters=TABLE_FILTERS)
        headers = fid_save.create_table('/',
                                        "header",
                                        IIDC_header_dtype,
                                        filters = TABLE_FILTERS)
        
        images._v_attrs['is_finished'] = 0
        
        desc = os.path.basename(fname)
        for nn, (header, img) in enumerate(tqdm.tqdm(mreader, desc = desc)):
            
            images[nn] = img
            headers.append(header)
        
        images._v_attrs['is_finished'] = 1
   
if __name__ == '__main__':
    import sys
    if sys.platform == 'darwin':
        root_dir = '/Volumes/Ext1/Data/AnalysisFingers/'
    else:
        root_dir = '/media/ver228/Ext1/Data/AnalysisFingers/'
    
    fnames = glob.glob(os.path.join(root_dir, '**', '*.movie'), recursive=True)
    fnames = sorted(fnames)
    
    for fname in tqdm.tqdm(fnames):
        movie_to_hdf5(fname)
    
    

    
    
        
