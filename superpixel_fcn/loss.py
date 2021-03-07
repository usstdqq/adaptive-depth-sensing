import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import poolfeat, upfeat

'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_pos_loss(prob_in, xy_feat, kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*(50+2)*H*W
    # output : B*9*H*W
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    prob = prob_in.clone()

    b, c, h, w = xy_feat.shape
    pooled_xy = poolfeat(xy_feat, prob, kernel_size, kernel_size)
    reconstr_xy_feat = upfeat(pooled_xy, prob, kernel_size, kernel_size)

    loss_map = reconstr_xy_feat - xy_feat # pos Bx2xHxW

    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b / S

    return loss_pos

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*(50+2)*H*W
    # output : B*9*H*W
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:] # pos Bx2xHxW

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum = (loss_sem + loss_pos)
    loss_sem_sum = loss_sem
    loss_pos_sum = loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum


def compute_color_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*(3+2)*H*W
    # output : B*9*H*W
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_pos_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:] # pos Bx2xHxW
    loss_color_map = reconstr_feat[:, :-2, :, :] - labxy_feat[:, :-2, :, :] # pos Bx3xHxW
    
    loss_color = torch.norm(loss_color_map, p=2, dim=1).sum() / b / S
    loss_pos = torch.norm(loss_pos_map, p=2, dim=1).sum() / b / S * m

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  (loss_color + loss_pos)
    loss_color_sum = loss_color
    loss_pos_sum = loss_pos

    return loss_sum, loss_color_sum,  loss_pos_sum
