from patch_function \
import rand_patch, \
makepatches_overlay, \
makepatches_overlay_normap, \
patchesback_overlay, \
find_ksp_andpadding, \
pad_back, \
pad_for_256, \
pad_back_256

import torch.utils.data as Data
import torch
import torch.nn.functional as F

import nibabel as nib
import numpy as np
import random
import os
from scipy import ndimage

def patch(x,y,block_size, slice_num):
    l = 0
    r = x.shape[1]
    if x.shape[2]>slice_num:
        if x.shape[0]<=block_size:
            start = np.random.randint(x.shape[2]-slice_num)
            x_p = x[:,:,start:start+slice_num,:]
            y_p = y[:,:,start:start+slice_num]
        else:
            start = np.random.randint([l, l, 0],[r-block_size, r-block_size, x.shape[2]-slice_num],3)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num]
    else:
        if x.shape[0]<=block_size:
            x_p = x
            y_p = y
        else:
            start = np.random.randint([l, l],[r-block_size, r-block_size],2)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
    return x_p, y_p

def pad_z(x, y, slice_num):
    if x.shape[2]<slice_num:
        x_new = np.zeros([x.shape[0], x.shape[1], slice_num, x.shape[3]])
        y_new = np.zeros([y.shape[0], y.shape[1], slice_num])
        x_new[:,:,0:x.shape[2],:] = x
        y_new[:,:,0:y.shape[2]] = y
    else:
        x_new = x
        y_new = y
    return x_new, y_new

def process(data_dir, biparametric=False, slice_num=64):
    #print(data_dir)    
    x = np.load(data_dir)['data']
    y = np.load(data_dir.replace('images','labels'))['data']
    x[np.where(np.isnan(x))] = 0
    y[np.where(np.isnan(y))] = 0
    inplane_size = 256
    #slice_num = 64
    
    t1c_mean = np.mean(x[:,:,:,0])
    t1c_std = np.std(x[:,:,:,0])    
    t1c = x[:,:,:,0]
    if biparametric:
        t2 = x[:,:,:,1]
        t2_mean = np.mean(x[:,:,:,1])
        t2_std = np.std(x[:,:,:,1])
    
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)    
        x = np.concatenate((t1c, t2),-1)
        x_p, y_p = patch(x, y, inplane_size, slice_num)
    else:
        x_p, y_p = patch(t1c, y, inplane_size, slice_num)
    
    flip_ratio = np.random.rand(1)
    if flip_ratio >= 0.5:
        x_p = np.fliplr(x_p)
        y_p = np.fliplr(y_p)
    rot_angle = np.random.randint(-15, 15)
    x_p = ndimage.rotate(x_p, rot_angle, reshape=False)
    y_p = ndimage.rotate(y_p, rot_angle, reshape=False)
    
    x_p, y_p = pad_z(x_p, y_p, slice_num)
    
    t1c_p = x_p[:,:,:,0]-t1c_mean
    t1c_p = t1c_p/t1c_std
    if biparametric:
        t2_p = x_p[:,:,:,1]-t2_mean
        t2_p = t2_p/t2_std   
    
    t1c_p = np.expand_dims(t1c_p, -1)
    if biparametric:
        t2_p = np.expand_dims(t2_p, -1)
        x_p = np.concatenate((t1c_p, t2_p),-1)
    else:
        x_p = t1c_p
        
    y_p[y_p>=0.5] = 1
    y_p[y_p<0.5] = 0
    
    #x_p = np.expand_dims(x_p, 0)
    y_p = np.expand_dims(y_p, -1)
    #y_p = np.expand_dims(y_p, 0)
    return np.array(x_p,dtype=np.float32), np.array(y_p,dtype=np.float32)

def tprocess(data_dir, biparametric=False, slice_num=64):
    #print(data_dir)
    x = np.load(data_dir)['data']    
    y = np.load(data_dir.replace('images','labels'))['data']
    x[np.where(np.isnan(x))] = 0
    y[np.where(np.isnan(y))] = 0
    inplane_size = 256
    #slice_num = 64
    
    t1c_mean = np.mean(x[:,:,:,0])
    t1c_std = np.std(x[:,:,:,0])    
    t1c = x[:,:,:,0]
    if biparametric:
        t2 = x[:,:,:,1]
        t2_mean = np.mean(x[:,:,:,1])
        t2_std = np.std(x[:,:,:,1])
    
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)    
        x = np.concatenate((t1c, t2),-1)
        x_p, y_p = patch(x, y, inplane_size, slice_num)
    else:
        x_p, y_p = patch(t1c, y, inplane_size, slice_num)
    
    x_p, y_p = pad_z(x_p, y_p, slice_num)
    
    t1c_p = x_p[:,:,:,0]-t1c_mean
    t1c_p = t1c_p/t1c_std
    if biparametric:
        t2_p = x_p[:,:,:,1]-t2_mean
        t2_p = t2_p/t2_std
    
    t1c_p = np.expand_dims(t1c_p, -1)
    if biparametric:
        t2_p = np.expand_dims(t2_p, -1)
        x_p = np.concatenate((t1c_p, t2_p),-1)
    else:
        x_p = t1c_p
        
    y_p[y_p>=0.5] = 1
    y_p[y_p<0.5] = 0
    
    #x_p = np.expand_dims(x_p, 0)
    y_p = np.expand_dims(y_p, -1)
    #y_p = np.expand_dims(y_p, 0)
    return np.array(x_p,dtype=np.float32), np.array(y_p,dtype=np.float32)

class Dataset(Data.Dataset):
    def __init__(self, list_, 
                 #random_t2=False, 
                 #only_t1c=False, 
                 augmentation=False,
                 biparametric=False,
                 slice_num=64):
        
            self.data_len = len(list_)
            self.list_ = list_            
            #self.random_t2 = random_t2
            self.slice_num = slice_num
            #self.only_t1c = only_t1c
            self.augmentation = augmentation
            self.biparametric = biparametric
            #print(biparametric)
            
    def __len__(self):        
        return self.data_len
    
    def __getitem__(self, index):
        img, gt = process(self.list_[index], self.biparametric, self.slice_num)
        
        img = img.transpose([3,0,1,2]) # C,H,W,D
        gt = gt.transpose([3,0,1,2]) # C,H,W,D        
        
        # to tensor format
        img = torch.from_numpy(np.double(img)).type(torch.FloatTensor)
        gt = torch.from_numpy(np.double(gt)).type(torch.int64)
        return img, gt


class Dataset_val(Data.Dataset):
    def __init__(self, list_, 
                 #no_T2=False,
                 #random_sample=False,
                 #only_t1c=False,
                 biparametric=False,
                 slice_num=64):
        
            self.data_len = len(list_)
            self.list_ = list_
            #self.no_T2 = no_T2
            #self.random_sample = random_sample
            #self.only_t1c = only_t1c
            self.slice_num = slice_num
            self.biparametric = biparametric
            
    def __len__(self):        
        return self.data_len
    
    def __getitem__(self, index):
        img, gt = tprocess(self.list_[index], self.biparametric, self.slice_num)
        
        img = img.transpose([3,0,1,2]) # C,H,W,D
        gt = gt.transpose([3,0,1,2]) # C,H,W,D
        
        # to tensor format
        img = torch.from_numpy(np.double(img)).type(torch.FloatTensor)
        gt = torch.from_numpy(np.double(gt)).type(torch.int64)
        return img, gt