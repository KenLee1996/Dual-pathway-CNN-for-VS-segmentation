import numpy as np

def mask2ind(mask):
    # this function make a binary mask into the linear indices
    # for binary mask input only

    # subscripts of the lesion pixel
    sub1 = np.argwhere(mask==1)
    
    # change the format
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for tmp in sub1: 
        tmp1.append(tmp[0])
        tmp2.append(tmp[1])
        tmp3.append(tmp[2])
    sub2 = tuple([np.array(tmp1),np.array(tmp2),np.array(tmp3)])
    

    if sub2[0].size!=0:
        # sub to ind
        indx = np.ravel_multi_index(sub2, mask.shape)
    else:
        indx = tuple([])
    
    # dictionary
    ind_dict = {'indx':indx, 'shape':mask.shape}
    
    return ind_dict

def ind2mask(ind_dict):
    # this function make indices into a binary mask
    
    sub = np.unravel_index(ind_dict['indx'], ind_dict['shape'])
    mask = np.zeros(ind_dict['shape']).astype('uint8')
    mask[sub] = 1    
    
    return mask