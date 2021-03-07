import os
import requests
import numpy as np
from torchvision import transforms
import torch

import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(real)-torch.log(fake)) ** 2 ) )
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    
    def forward(self, fake, real):
        loss = torch.mean( torch.abs(10.*real-10.*fake) )
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()
    
    def forward(self, fake, real):
        loss = torch.mean( torch.abs(torch.log(real)-torch.log(fake)) )
        return loss