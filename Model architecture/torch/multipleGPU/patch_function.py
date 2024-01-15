import torch
import torch.nn.functional as F
import numpy as np


def rand_patch(x,y,block_size,slice_num=24):
    #block_size = 176
    x = x[:,:,:,:] # h, w, d, c
    #slice_num = 24
    l = 0
    r = x.shape[1]
    if x.shape[2]>slice_num:
        if x.shape[0]<=block_size:
            start = np.random.randint(x.shape[2]-slice_num)
            x_p = x[:,:,start:start+slice_num]
            y_p = y[:,:,start:start+slice_num]
        else:
            start = np.random.randint([l, l, 0],[r-block_size, r-block_size, x.shape[2]-slice_num],3)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num]
    elif x.shape[2]==slice_num:
        if x.shape[0]<=block_size:
            x_p = x
            y_p = y
        else:
            start = np.random.randint([l, l],[r-block_size, r-block_size],2)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
    elif x.shape[2]<slice_num:
        if x.shape[0]<=block_size:
            x_p = x
            y_p = y
        else:
            start = np.random.randint([l, l],[r-block_size, r-block_size],2)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
            
        tmp = np.zeros((x_p.shape[0],x_p.shape[1],slice_num-x_p.shape[2],x_p.shape[3]))
        x_p = np.concatenate((x_p, tmp), axis=2)
        tmp = np.zeros((y_p.shape[0],y_p.shape[1],slice_num-y_p.shape[2]))
        y_p = np.concatenate((y_p, tmp), axis=2)
    
    return x_p, y_p



def makepatches_overlay(threeD_img, 
                        kernel_size=(256,256,20), 
                        stride=(256,256,20),
                        ):
    
    #if torch.is_tensor(threeD_img)==0:
        #threeD_img = torch.from_numpy(np.double(threeD_img)).type(torch.FloatTensor)
    
    # pad to ensure the whole img can be inference
    pad_whole = stride-((np.array(threeD_img.shape[-3:])-np.array(kernel_size))%stride)
    pad_whole = pad_whole%stride
    # in case that the dimension of 3dimg less than kernel size and is multiple for stride
    pad_add = np.array(threeD_img.shape[-3:])+pad_whole-kernel_size 
    pad_whole[pad_add<0] = pad_whole[pad_add<0]+np.abs(pad_add[pad_add<0]) 
    pad_a = np.ceil(pad_whole/2).astype('int')
    pad_b = np.floor(pad_whole/2).astype('int')
    p3d = (pad_b[2],pad_a[2],pad_b[1],pad_a[1],pad_b[0],pad_a[0])
    threeD_img = F.pad(threeD_img,p3d)
    
    # input should be in torch tensor
    
    # B, C, H, W, D  -> B, C, D, H, W
    threeD_img = threeD_img.permute((0,1,4,2,3)) #transpose for depth
    
    # threeD_img->B, C, D, H, W    
    B, C, D, H, W = threeD_img.shape 

    patches = threeD_img.unfold(2, kernel_size[2], stride[2])\
                .unfold(3, kernel_size[0], stride[0])\
                .unfold(4, kernel_size[1], stride[1])
    # [B, C, nb_patches_d, nb_patches_h, nb_patches_w, kernel_size, kernel_size, kernel_size]
    
    patches = patches.contiguous()\
                .view(B, C, -1, kernel_size[2], kernel_size[0], kernel_size[1])\
                .permute(2,0,1,3,4,5)\
                .view(-1, C, kernel_size[2], kernel_size[0], kernel_size[1])
    #print(patches.shape) 
    # [nb_patches_all*B, C, kernel_size, kernel_size, kernel_size]
    
    # padding
    #p3d = (2, 2, 2, 2, 2, 2)
    #patches = F.pad(patches,p3d)
    
    # B, C, D, H, W  -> B, C, H, W, D
    patches = patches.permute((0,1,3,4,2)) #transpose for depth
    
    return patches


def makepatches_overlay_normap(threeD_img_dim, 
                               kernel_size=(256,256,20), 
                               stride=(256,256,20)):
    
    # pad to ensure the whole img can be inference
    pad_whole = stride-((np.array(threeD_img_dim[-3:])-np.array(kernel_size))%stride)
    pad_whole = pad_whole%stride
    # in case that the dimension of 3dimg less than kernel size and is multiple for stride
    pad_add = np.array(threeD_img_dim[-3:])+pad_whole-kernel_size 
    pad_whole[pad_add<0] = pad_whole[pad_add<0]+np.abs(pad_add[pad_add<0]) 
    
    # threeD_img_dim->B, C, H, W, D
    B, C, H, W, D = threeD_img_dim
    
    nor_map = torch.ones(B, C, D+pad_whole[2], H+pad_whole[0], W+pad_whole[1])
    
    # map for normalization
    nor_map = nor_map.unfold(2, kernel_size[2], stride[2])\
                .unfold(3, kernel_size[0], stride[0])\
                .unfold(4, kernel_size[1], stride[1])
    nor_map = nor_map.contiguous()\
                .view(B, C, -1, kernel_size[2], kernel_size[0], kernel_size[1])\
                .permute(2,0,1,3,4,5)\
                .view(-1, C, kernel_size[2], kernel_size[0], kernel_size[1])
    
    # B, C, D, H, W  -> B, C, H, W, D
    nor_map = nor_map.permute((0,1,3,4,2)) #transpose for depth
    
    return nor_map

def patchesback_overlay(patches, nor_map, threeD_img_dim, 
                        kernel_size=(256,256,20), stride=(256,256,20)):
    
    #if torch.is_tensor(patches)==0:
        #patches = torch.from_numpy(np.double(patches)).type(torch.FloatTensor)
        
    # pad information
    pad_whole = stride-((np.array(threeD_img_dim[-3:])-np.array(kernel_size))%stride)
    pad_whole = pad_whole%stride
    # in case that the dimension of 3dimg less than kernel size and is multiple for stride
    pad_add = np.array(threeD_img_dim[-3:])+pad_whole-kernel_size 
    pad_whole[pad_add<0] = pad_whole[pad_add<0]+np.abs(pad_add[pad_add<0])
    pad_a = np.ceil(pad_whole/2).astype('int')
    pad_b = np.floor(pad_whole/2).astype('int')
    
    threeD_img_dim_pad = np.array(threeD_img_dim)+np.array([0,0,*pad_whole])
        
    # input patches->B, P, C, Hh, Ww, Dd   
    
    # permute
    patches = patches.permute((0,5,1,2,3,4))
    nor_map = nor_map.permute((0,5,1,2,3,4))
    # patches->B, Dd, P, C, Hh, Ww
    B, Dd, P, C, Hh, Ww = patches.shape
    
    # threeD_img->B, C, H, W, D
    B, C, H, W, D = threeD_img_dim_pad
    Pd = int((D-(kernel_size[2]-1)-1)/stride[2]+1)
    Phw = int(P/Pd)
    
    # reshape and permute
    patches = patches.contiguous().view(-1, Phw, C, kernel_size[0], kernel_size[1])\
                .permute(0,2,1,3,4)
    nor_map = nor_map.contiguous().view(-1, Phw, C, kernel_size[0], kernel_size[1])\
                .permute(0,2,1,3,4)
    # patches->B*Dd*Pd, C, Phw, H, W
    
    # reshape output to match F.fold input for H and W
    patches = patches.contiguous().view(B*Dd*Pd, C, Phw, kernel_size[0]*kernel_size[1])
    nor_map = nor_map.contiguous().view(B*Dd*Pd, C, Phw, kernel_size[0]*kernel_size[1]) 
    # patches->B*Dd*Pd, C, Phw, H*W
    
    # permute
    patches = patches.permute(0, 1, 3, 2)
    nor_map = nor_map.permute(0, 1, 3, 2)
    # patches->B*Dd*Pd, C, H*W, Phw
    
    # reshape output to match F.fold input for H and W
    patches = patches.contiguous().view(B*Dd*Pd, C*kernel_size[0]*kernel_size[1], Phw)
    nor_map = nor_map.contiguous().view(B*Dd*Pd, C*kernel_size[0]*kernel_size[1], Phw)
    # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    # fold for H and W
    threeD_img = F.fold(
        patches, output_size=(H, W), kernel_size=kernel_size[0:2], stride=stride[0:2])    
    nor_map = F.fold(
        nor_map, output_size=(H, W), kernel_size=kernel_size[0:2], stride=stride[0:2])    
    # B*Dd*Pd, C, H, W
    
    # reshape output to match F.fold input for D
    threeD_img = threeD_img.contiguous().view(B,Dd,Pd,C,H,W).permute(0,4,5,3,1,2).contiguous().view(-1,C*Dd,Pd)
    nor_map = nor_map.contiguous().view(B,Dd,Pd,C,H,W).permute(0,4,5,3,1,2).contiguous().view(-1,C*Dd,Pd)
    # B*H*W,C*Dd,Pd
    
    # fold for D
    threeD_img = F.fold(
        threeD_img, output_size=(1,D), kernel_size=(1,kernel_size[2]), stride=(1,stride[2]))
    nor_map = F.fold(
        nor_map, output_size=(1,D), kernel_size=(1,kernel_size[2]), stride=(1,stride[2]))
    # B*H*W,C,1,D
    
    # reshape and permute back
    threeD_img = threeD_img.contiguous().view(B,H,W,C,D).permute(0,3,1,2,4)
    nor_map = nor_map.contiguous().view(B,H,W,C,D).permute(0,3,1,2,4)
    
    # normalize (B,C,H,W,D)
    threeD_img = threeD_img/nor_map    
    
    B, C, H, W, D = threeD_img_dim_pad    
    threeD_img = threeD_img[:,:,
                            pad_b[0]:H-pad_a[0],
                            pad_b[1]:W-pad_a[1],
                            pad_b[2]:D-pad_a[2]]
    
    
    return threeD_img


def find_ksp_andpadding(img,shape3d):
    # input in H,W,D
    H, W, D = shape3d
    
    # find D
    D2 = D
    num_d = 2
    pad_d = 0
    while D/num_d>20:
        num_d = num_d+1
    while D2%num_d!=0:
        D2 = D2+1
        pad_d = pad_d+1
    # find H
    H2 = H
    num_h = 2
    pad_h = 0
    while H/num_h>256:
        num_h = num_h+1
    while H2%num_h!=0:
        H2 = H2+1
        pad_h = pad_h+1
    # find W
    W2 = W
    num_w = 2
    pad_w = 0
    while W/num_w>256:
        num_w = num_w+1
    while W2%num_w!=0:
        W2 = W2+1
        pad_w = pad_w+1
    
    padded_shape = np.array([H2,W2,D2]).astype('int')
    padding = np.array([pad_h,pad_w,pad_d]).astype('int')
    kands = np.array([H2/num_h,W2/num_w,D2/num_d]).astype('int')
    
    # padding
    p3d = (0, padding[2], 0, padding[1], 0, padding[0]) #(uD,lD,uW,lW,uH,lH)
    img_pad = F.pad(img,p3d)
    
        
    return img_pad, padding, kands


def pad_back(padded_img, padding):
    
    # input img -> B, C, H, W, D 
    B, C, H, W, D = padded_img.shape
    
    # unpad
    unpadded_img = padded_img[:,:,0:H-padding[0],0:W-padding[1],0:D-padding[2]]
    
    return unpadded_img


def pad_for_256(img):
    
    # img -> B, C, H, W, D
    B, C, H, W, D = img.shape
    
    if H!=256 and H<256:
        pad_h = 256-H
        p3d = (0,0,0,0,0,pad_h)
        img = F.pad(img,p3d)
    else:
        pad_h = 0
    if W!=256 and W<256:
        pad_w = 256-W
        p3d = (0,0,0,pad_w,0,0)
        img = F.pad(img,p3d)
    else:
        pad_w = 0
        
    return img, (pad_h,pad_w,0)


def pad_back_256(padded_img, padding):
    
    # input img -> B, C, H, W, D 
    B, C, H, W, D = padded_img.shape
    
    # unpad
    unpadded_img = padded_img[:,:,0:H-padding[0],0:W-padding[1],0:D-padding[2]]
    
    return unpadded_img


def patch_inference(**kwargs):

    # input
    model = kwargs['model']
    batch_x = kwargs['batch_x']
    kernel_size = kwargs['kernel_size']
    stride = kwargs['stride']
    inf_patch_num = kwargs['inf_pnum']

    # make patches
    patches = makepatches_overlay(batch_x, 
                        kernel_size=kernel_size, 
                        stride=stride)           
    
    itr_num = (patches.shape[0]-(patches.shape[0]%inf_patch_num))/inf_patch_num
    itr_num = int(itr_num)
    
    with torch.no_grad():
        model.to(torch.device("cuda"))
        model.train(False)
        # patch inference
        for i in range(itr_num+1):
            if i != itr_num:
                output_p = model(patches[i*inf_patch_num:(i+1)*inf_patch_num,
                                         :,:,:,:].cuda())
            else:
                output_p = model(patches[i*inf_patch_num:,
                                         :,:,:,:].cuda())
            if i == 0:
                output = output_p.cpu()
            else:
                output = torch.cat((output,output_p.cpu()),dim=0)
        
    # normalized map for overlap
    nmap = makepatches_overlay_normap(batch_x.shape, 
                        kernel_size=kernel_size, 
                        stride=stride)
    
    # patch back
    output = patchesback_overlay(output.view(-1,*output.shape), 
                    nmap.view(-1,*output.shape), 
                    batch_x.shape, 
                    kernel_size=kernel_size, 
                    stride=stride)

    return output
