import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from retinanet.models import RetinaNet, FocalLoss
from retinanet.flow import VesicleBBFlow, collate_fn

from torch.utils.data import DataLoader 

import torch
#%%
if __name__ == '__main__':
    num_classes = 2
    gen = VesicleBBFlow(is_clean_data = True)
    criterion = FocalLoss(num_classes)
    
    loader = DataLoader(gen, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    for img, target in loader:
        break
    #%% 
    
    mod = RetinaNet(num_classes=num_classes, num_anchors = gen.encoder.n_anchors_shapes)
#    
#    loc_preds, cls_preds = mod(img)
#    
#    loc_grads = torch.randn(loc_preds.size())
#    loc_preds.backward(loc_grads)
#    
#    loc_preds, cls_preds = mod(img)
#    cls_grads = torch.randn(cls_preds.size())
#    cls_preds.backward(cls_grads)
    #%%
    cls_preds, loc_preds = mod(img)
    
    #cls_preds = torch.zeros_like(cls_preds)
    clf_loss, loc_loss = criterion((cls_preds, loc_preds), target)
    #%%
    loss = clf_loss + loc_loss
    
    loss.backward()
