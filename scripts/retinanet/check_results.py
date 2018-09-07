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


from retinanet.flow import VesicleBBFlow, collate_fn
from retinanet.models import RetinaNet, FocalLoss
from retinanet.trainer import log_dir_root


import tqdm
import torch
from torch.utils.data import DataLoader
#%%
def nms(dets, scores, overlap_thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return keep
#%%
if __name__ == '__main__':
    batch_size = 1
    num_classes = 2
    
    
    #model_bn = 'adam_20180905_180020_retinanet-resnet34_adam_lr1e-05_batch16' 
    
    model_bn = 'adam_20180906_112815_retinanet-resnet34_clean_adam_lr0.0001_batch16'
    #model_bn = 'adam_20180906_111632_retinanet-resnet34_adam_lr0.0001_batch16'
    backbone = model_bn.partition('retinanet-')[-1].partition('_')[0]
    
    #model_path = log_dir_root / model_bn / 'model_best.pth.tar'
    model_path = log_dir_root / model_bn / 'checkpoint.pth.tar'
    
    
    
    
    #%%
    gen = VesicleBBFlow(is_clean_data = False)
    
    
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        collate_fn = collate_fn,
                        )
    #%%
    model = RetinaNet(num_classes = num_classes, 
                      num_anchors = gen.encoder.n_anchors_shapes,
                      backbone = backbone)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    criterion = FocalLoss(num_classes)
    #%%
    gen.test()
    
    tot_loss = 0
    n_batches = 0
    for X, (clf_target, loc_target) in tqdm.tqdm(loader):
        with torch.no_grad():
            clf_preds, loc_preds = model(X)
            
            clf_loss, loc_loss = criterion((clf_preds, loc_preds), (clf_target, loc_target))
            loss = clf_loss + loc_loss
            
            tot_loss += loss
        
        
        out_loc = loc_preds[0].cpu().detach().numpy()
        scores, out_clf = clf_preds[0].cpu().detach().max(-1)
        
        
        bad = scores<0.25
        
        out_clf += 1
        out_clf[bad] = 0
        out_clf = out_clf.numpy()
        
        #out_clf = clf_target[0].cpu().detach().numpy()
        #out_loc = loc_target[0].cpu().detach().numpy()
        
        pos_ = out_clf>0
        closest_boxes = gen.encoder._anc_xy[pos_]
        out_clf, out_loc = gen.encoder.decode(out_clf, out_loc)
        
        scores = scores[~bad].numpy()
        
        
        keep = nms(out_loc, scores, 0.1)
        
        out_clf, out_loc = out_clf[keep], out_loc[keep]
        
        #if out_clf.sum()==0:
        #    continue
        
        fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
        
        axs.imshow(X.squeeze())
        for lab, (x0, y0, x1, y1) in zip(out_clf, out_loc):
            c = 'y' if lab == 1 else 'r'
            axs.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), c)
        
#        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
#        
#        axs[0].imshow(X.squeeze())
#        for lab, (x0, y0, x1, y1) in zip(out_clf, out_loc):
#            c = 'y' if lab == 1 else 'r'
#            axs[0].plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), c)
#            
#        
#        axs[1].imshow(X.squeeze())
#        for (x0, y0, x1, y1) in closest_boxes:
#            axs[1].plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'c:')
        n_batches +=1
        #if n_batches > 10:
        #    break
        
    tot_loss = tot_loss/n_batches
    #%%
    img = X.squeeze().numpy()

    L = out_loc[:, 2:]-out_loc[:, :2]
    cm = out_loc[:, :2] + L/2
    
    center_rot = tuple(x//2 for x in img.shape[::-1])
    rot_angle = random.randint(-180, 180)
    
    M = cv2.getRotationMatrix2D(center_rot, rot_angle, 1)
    img_r = cv2.warpAffine(img, M, img.shape)
    
    plt.figure()
    plt.imshow(img, vmin=0)
    plt.plot(cm[:, 0], cm[:, 1], 'xr')
    for (x0, y0, x1, y1) in out_loc:
        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')
    
    cm_r = np.matmul(np.concatenate((cm, np.ones((cm.shape[0],1))), axis=1), M.T)
    #I can do this because this data is suppose to be circles not squares
    out_loc_r = np.concatenate((cm_r-L/2, cm_r+L/2), axis=1)
    
    plt.figure()
    plt.imshow(img_r)
    plt.plot(cm_r[:, 0], cm_r[:, 1], 'xr')
    for (x0, y0, x1, y1) in out_loc_r:
        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')