#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:11:37 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from retinanet.models import RetinaNet, nms
from retinanet.flow import BoxEncoder, get_jaccard_index
from retinanet.trainer import log_dir_root

from read_movies.moviereader import MovieReader

import tqdm
import torch

#%%
import tables
import cv2
import numpy as np
import math


TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)

def _init_table(traj_fid):
    
    fields2remove = ['/trajectories_data', '/timeseries_data', '/blobs_data']
    for field in fields2remove:
        if field in traj_fid:
            traj_fid.remove_node(field)
    # intialize main table
    int_dtypes = [('worm_index_joined', np.int),
                  ('frame_number', np.int),
                  ('skeleton_id', np.int)
                  ]
    dd = ['coord_x', 'coord_y', 'threshold', 'roi_size', 'worm_index_manual', 'worm_label']
    
    float32_dtypes = [(x, np.float32) for x in dd]
    
    plate_worms_dtype = np.dtype(int_dtypes + float32_dtypes)
    trajectories_data = traj_fid.create_table('/',
                                        "trajectories_data",
                                        plate_worms_dtype,
                                        filters = TABLE_FILTERS)
    
    

    return trajectories_data    

#%%

#%%
if __name__ == '__main__':
    model_bn = 'adam_20180907_004840_retinanet-resnet50_adam_lr0.0001_batch16'
    backbone = model_bn.partition('retinanet-')[-1].partition('_')[0]
    
    model_path = log_dir_root / model_bn / 'model_best.pth.tar'
    #model_path = log_dir_root / model_bn / 'checkpoint.pth.tar'
    
    batch_size = 4
    num_classes = 2
    
    scale = (7, 11.1)
    zoom_ = 2.
    score_th = 0.25
    
    _is_debug = True
    #%%
    model = RetinaNet(num_classes = num_classes, 
                      num_anchors = 3,
                      backbone = backbone)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/data/ramp100.22Sep2016_17.49.11.movie'
    #movie_name =  '/Users/avelinojaver/OneDrive - Nexus365/vesicle/data/script_ramp.08Dec2015_16.45.56.movie'
    movie_name = '/Users/avelinojaver/OneDrive - Nexus365/vesicle/data/ramp100.29Oct2015_17.54.52.movie'
    save_name = movie_name.replace('/data/', '/cleaned/').replace('.movie', '.hdf5')
    
    
    reader = MovieReader(str(movie_name))
    tot = len(reader)
    img_h, img_w = reader.height, reader.width
    
    shape2resize = (int(img_h/zoom_), int(img_w/zoom_))
    nn = 2**5
    ss = [math.ceil(x/nn)*nn - x for x in shape2resize]
    pad_size = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
    
    final_size = [x + s for x,s in zip(shape2resize, ss)]
    encoder = BoxEncoder(final_size)
    
    batch = []
    traj_dat = []
    
    
    prev_out_loc = None
    index_prev = None
    current_traj_id = 0
    
    for current_frame, (_, img_o) in enumerate(tqdm.tqdm(reader)):
        img = cv2.resize(img_o, shape2resize[::-1])
        img = (np.log(img)-scale[0])/(scale[1]-scale[0])
        img = np.pad(img, pad_size, mode='constant')
        
        batch.append(img[None])
        
        
        
                
        if len(batch) >= batch_size or current_frame == tot-1:
            batch_r = np.stack(batch)
            X = torch.from_numpy(batch_r)
            batch = []
            
            clf_preds, loc_preds = model(X)
            #%%
            for i_offset, (loc_pred, clf_pred) in enumerate(zip(loc_preds, clf_preds)):
                
                out_loc = loc_pred.cpu().detach().numpy()
                
                
                scores, out_clf = clf_pred.cpu().detach().max(-1)
                
                bad = scores< score_th
                    
                out_clf += 1
                out_clf[bad] = 0
                out_clf = out_clf.numpy()
                
                pos_ = out_clf>0
        
                closest_boxes = encoder._anc_xy[pos_]
                out_clf, out_loc = encoder.decode(out_clf, out_loc)
                
                scores = scores[~bad].numpy()
                
                keep = nms(out_loc, scores, 0.1)
                
                out_clf, out_loc = out_clf[keep], out_loc[keep]
                
                out_loc[..., 0] = (out_loc[..., 0] - pad_size[1][0])*img_h/shape2resize[0]
                out_loc[..., 2] = (out_loc[..., 2] - pad_size[1][0])*img_h/shape2resize[0]
                
                out_loc[..., 1] = (out_loc[..., 1] - pad_size[0][0])*img_w/shape2resize[1]
                out_loc[..., 3] = (out_loc[..., 3] - pad_size[0][0])*img_w/shape2resize[1]
                
    
                roi_size = np.abs(out_loc[..., 2:] - out_loc[..., :2]).mean(axis=1)
                coord_x = (out_loc[..., 0] + out_loc[..., 2])/2
                coord_y = (out_loc[..., 1] + out_loc[..., 3])/2
                
                frame_number = current_frame - batch_r.shape[0] + (i_offset + 1)
                
                index_current = list(range(current_traj_id+1, current_traj_id+out_loc.shape[0]+1))
                current_traj_id += len(index_current)
                
                if prev_out_loc is not None:
                    jaccard_ind = get_jaccard_index(prev_out_loc, out_loc)
                    
                    prev_intersect_f = [np.where(x>0.5)[0] for x in jaccard_ind]
                    simple_pairs = [(i_prev, vv[0]) for i_prev, vv in enumerate(prev_intersect_f) if len(vv) ==1]
                    
                    
                    for i1, i2 in simple_pairs:
                        index_current[i2] = index_prev[i1]
                
                for lab, rs, cx, cy, ind in zip(out_clf, roi_size, coord_x, coord_y, index_current):
                    
                    row_data = (ind, frame_number, -1, cx, cy, -1, rs, ind, lab)
                    traj_dat.append(row_data)
                    #%%
                    #%%
                
                prev_out_loc = out_loc
                index_prev = index_current
            #%%
            if _is_debug:
                fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
    
                axs.imshow(img_o)
                for lab, (x0, y0, x1, y1) in zip(out_clf, out_loc):
                    c = 'y' if lab == 1 else 'r'
                    axs.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), c)
                
               #%% 
                if len(plt.get_fignums()) > 0:
                    break                        
                        
    with tables.File(save_name, 'r+') as fid:
        trajectories_data = _init_table(fid)
        trajectories_data.append(traj_dat)

            
        
    #%%

    
    
    