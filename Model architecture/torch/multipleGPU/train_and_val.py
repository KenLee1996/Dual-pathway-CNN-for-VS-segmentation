# -*- coding: utf-8 -*-
import time
import torch
from patch_function import makepatches_overlay, \
makepatches_overlay_normap, \
patchesback_overlay, \
find_ksp_andpadding, \
pad_back, \
pad_for_256, \
pad_back_256, \
patch_inference
import numpy as np
import scipy.io as sio
import os
import time
from maskindprocess import mask2ind


#訓練模式
def train(train_loader,model,loss_fun,optimizer,epoch,path):
    
    f = open(path + '/loss_curve.txt', 'a')
    
    model.train() # Turn on the train mode
    

    total_loss = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    
    start_time = time.time()

    for step, (batch_x, batch_y) in enumerate(train_loader):
        
        # reset gradient of the optimizer
        optimizer.zero_grad()
                
        # apply GPU setup
        batch_x = batch_x#.cuda() # data
        batch_y = batch_y#.cuda() # ground truth
        
        # model prediction
        output = model(batch_x)
        
        # loss calculation
        loss_dice = loss_fun[0](output,batch_y)
        loss_gdl = loss_fun[1](output,batch_y)
        dice_score = loss_fun[2](output,batch_y)
        
        # backward and update model weights
        #loss_dice.backward()    
        loss_gdl.backward()    
        optimizer.step()        
        
        
        # print log
        total_loss += loss_dice.item()
        total_loss2 += loss_gdl.item()
        total_loss3 += dice_score.mean().item()
        log_interval = 100
        
        if (step+1) % log_interval == 0 or step+1 == len(train_loader):
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  ' ms/batch {:5.2f} | '
                  ' loss_dice {:5.2f}  | '
                  ' loss_gdl {:5.2f}  | '
                  ' dice_score {:5.2f}  | '.format(
                      epoch, step+1, len(train_loader),
                      elapsed * 1000,
                      total_loss/(step+1),
                      total_loss2/(step+1),
                      total_loss3/(step+1)))
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                    ' ms/batch {:5.2f} | '
                    ' loss_dice {:5.2f}  | '
                    ' loss_gdl {:5.2f}  | '.format(
                        epoch, step+1, len(train_loader), 
                        elapsed * 1000,  
                        total_loss/(step+1),
                        total_loss2/(step+1),
                        total_loss3/(step+1))+'\r\n')
        
            
            start_time = time.time()
        
    del batch_x, batch_y
    torch.cuda.empty_cache()
        
    f.close()
    
    return total_loss2    




# 每個epoch完的測試模式
def validation(val_loader,model,loss_fun,epoch,path):
    f = open(path + '/loss_curve.txt', 'a')
    
    with torch.no_grad():
        model.to(torch.device("cuda"))
        model.train(False)
    

    total_loss = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    start_time = time.time()
    for step, (batch_x, batch_y) in enumerate(val_loader):
        
        with torch.no_grad():

            # apply GPU setup
            batch_x = batch_x#.cuda() # data
            batch_y = batch_y#.cuda() # ground truth

            # model prediction
            output = model(batch_x)

            # loss calculation
            loss_dice = loss_fun[0](output,batch_y)
            loss_gdl = loss_fun[1](output,batch_y)
            dice_score = loss_fun[2](output,batch_y)


            # print log
            total_loss += loss_dice.item()
            total_loss2 += loss_gdl.item()
            total_loss3 += dice_score.mean().item()
            log_interval = 100

            if (step+1) % log_interval == 0 or step+1 == len(val_loader):
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      ' ms/batch {:5.2f} | '
                      ' loss_dice {:5.2f}  | '
                      ' loss_gdl {:5.2f}  | '
                      ' dice_score {:5.2f}  | '.format(
                          epoch, step+1, len(val_loader),
                          elapsed * 1000,
                          total_loss/(step+1),
                          total_loss2/(step+1),
                          total_loss3/(step+1)))
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                        ' ms/batch {:5.2f} | '
                        ' loss_dice {:5.2f}  | '
                        ' loss_gdl {:5.2f}  | '.format(
                            epoch, step+1, len(val_loader), 
                            elapsed * 1000,  
                            total_loss/(step+1),
                            total_loss2/(step+1),
                            total_loss3/(step+1))+'\r\n')


                start_time = time.time()
   
    del batch_x, batch_y
    torch.cuda.empty_cache()
        
    f.close()
            
    return total_loss2