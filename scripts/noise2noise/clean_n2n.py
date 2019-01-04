9#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:08:14 2018

@author: avelinojaver
"""
import tables
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from noise2noise.models import UNet
from noise2noise.trainer import log_dir_root
from read_movies.moviereader import MovieReader

import torch
import numpy as np
import tqdm
from pathlib import Path

def _clean_hdf5_movie(fname_ori, fname_new, model, scale_log):
    
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='blosc',
        shuffle=True,
        fletcher32=True)
    
    reader = MovieReader(str(fname_ori))
    tot = len(reader)
    height = reader.height
    width = reader.width
    with tables.File(str(fname_new), 'w') as fid_new:
        block = fid_new.create_carray(
                                '/',
                                'mask',
                                #tables.Float32Atom(),
                                tables.UInt16Atom(),
                                shape = (tot, height, width),
                                chunkshape = (1, height, width),
                                filters=TABLE_FILTERS)
        
        batch = []
        for ii, (header, img) in enumerate(tqdm.tqdm(reader)):
            batch.append(img[None].astype(np.float32))
            
            if len(batch) >= batch_size or ii == (tot-1):
                
        
                with torch.no_grad():
                    Xv = torch.from_numpy(np.array(batch))
                    batch = []
                    
                    Xv = Xv.to(device)
                    Xv.log_().add_(-scale_log[0]).div_(scale_log[1]-scale_log[0])
                    
                    
                    Xhat = model(Xv)
                    Xhat.mul_(scale_log[1]-scale_log[0]).add_(scale_log[0]).exp_()
                    
                    
                    ss = Xhat.shape
                    xhat = Xhat.cpu().detach().view(ss[0], ss[2], ss[3]).numpy()
                    xhat = xhat.round().clip(0, 2**16-1).astype(np.uint16)
            
            
                    block[ii-xhat.shape[0]+1:ii+1] = xhat


if __name__ == '__main__':
    cuda_id = 0
    batch_size = 2
    
    scale_log = (7, 11.1)
    
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/microglia/hdf_movies/movies/2018.08.22_movies/180822_MicVid_20X_Dispense/180822_MicVid_20X_Dispense_J9-Media_J11-100uMATP_1/')
    save_dir = Path.home() / 'workspace/Vesicles/movies_cleaned/'
    root_dir = Path.home() / 'workspace/Vesicles/data/'
    
    
    
    
    model_path = log_dir_root / 'l1_20180819_122435_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
        
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(str(model_path), map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = model.to(device)
    model.eval()
    
    fnames = list(root_dir.rglob('*.movie'))
    
    for fname_ori in tqdm.tqdm(fnames):
        if fname_ori.name.startswith('Clean_'):
            continue
        
        new_dname = Path(str(fname_ori.parent).replace(str(root_dir), str(save_dir)))
        new_dname.mkdir(parents=True, exist_ok=True)
        fname_new = new_dname / (fname_ori.stem + '.hdf5')
        
        
        _clean_hdf5_movie(fname_ori, fname_new, model, scale_log = scale_log)
    