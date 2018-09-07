import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

import tqdm

from retinanet.flow import VesicleBBFlow, collate_fn
from torch.utils.data import DataLoader 
#%%
if __name__ == '__main__':
    batch_size = 16
    gen = VesicleBBFlow(is_clean_data = True)
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        collate_fn = collate_fn)
#    for kk in tqdm.trange(100):
#        for X, (clf_target, loc_target) in tqdm.tqdm(loader):
#            pass
#            break
#        break
        #%%
    gen.test()
    for ii in range(10, 20):
        img, clf_target, loc_target = gen[ii]
        
        
        out_lab, out_loc = gen.encoder.decode(clf_target, loc_target)
        
        
        anchors_xy_r = gen.encoder._anc_xy.reshape((-1,4))
        pos_ = clf_target.reshape(-1)>0
        closest_boxes = anchors_xy_r[pos_]
        
        plt.figure()
        plt.imshow(img.squeeze(), cmap='gray')
        
#        for lab, (x0, y0, x1, y1) in zip(out_lab, closest_boxes):
#            c = 'c' if lab == 1 else 'b'
#            plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), c+':')
        
        for lab, (x0, y0, x1, y1) in zip(out_lab, out_loc):
            c = 'y' if lab == 1 else 'r'
            plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), c)
        
        
        plt.xlim([0, gen.crop_size-1])
        plt.ylim([0, gen.crop_size-1])
        plt.title(ii)
        
        #%%
#    img, bboxes, img_ori, ori_coords, M = gen[3]
#    #%%
#    
#    
#    plt.figure()
#    plt.imshow(img.squeeze(), cmap='gray')
#    for (x0, y0, x1, y1) in bboxes:
#        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')
#    plt.axis('equal')
#    
#    
#    #%%
#    plt.figure()
#    plt.imshow(img_ori.squeeze(), cmap='gray')
#    for (x0, y0, x1, y1) in ori_coords[['x0', 'y0', 'x1', 'y1']].values:
#        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')
#    plt.axis('equal')
#    
#    #%%
#    import cv2
#    img = img_ori.squeeze()
#    vrows = ori_coords.copy()
#    
#    pad_size = round((math.sqrt(2)-1)*gen.crop_size/2)
#    crop_size_padded = 2*pad_size + gen.crop_size
#    crop_left, crop_right = int(math.ceil(crop_size_padded/2)), int(math.floor(crop_size_padded/2))
#    center_rot = (crop_left, crop_right)
#    
#    img, vrows = gen._random_crop(img, vrows, _crop_size = crop_size_padded)
#    
#    plt.figure()
#    plt.imshow(img.squeeze(), cmap='gray')
#    for (x0, y0, x1, y1) in vrows[['x0', 'y0', 'x1', 'y1']].values:
#        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')
#    plt.axis('equal')
#    #%%  
#    rot_angle = random.randint(-180, 180)
#    M = cv2.getRotationMatrix2D(center_rot, rot_angle, 1)
#    
#    cc = (crop_size_padded, crop_size_padded)
#    img_r = cv2.warpAffine(img, M, cc)
#    img_r = img_r[pad_size:-pad_size, pad_size:-pad_size]
#    
#    vrows_r = vrows.copy()
#    R = vrows_r['bb_size']/2
#    cm = (vrows_r[['x0','y0']].values + R[:,None])
#    
#    cm_r = np.matmul(np.pad(cm, ((0,0), (0,1)), 'constant', constant_values=1), M.T)
#    
#    vrows_r['x0'] = cm_r[:,0] - R
#    vrows_r['y0'] = cm_r[:,1] - R
#    vrows_r['x1'] = cm_r[:,0] + R
#    vrows_r['y1'] = cm_r[:,1] + R
#    
#    vrows_r[['x0', 'y0', 'x1', 'y1']] -= pad_size 
#    
#    plt.figure()
#    plt.imshow(img_r.squeeze(), cmap='gray')
#    for (x0, y0, x1, y1) in vrows_r[['x0', 'y0', 'x1', 'y1']].values:
#        plt.plot((x0,x1, x1,x0,x0),(y0,y0,y1,y1,y0), 'r')
#    plt.axis('equal')