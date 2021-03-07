from __future__ import print_function

import os
import sys
THIS_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_PATH, "superpixel_fcn"))

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloaders.data_utils import KITTI_Dataset
from tensorboard_logger import configure, log_value, log_images

from superpixel_fcn.loss import compute_color_pos_loss
from superpixel_fcn.train_util import init_spixel_grid, poolfeat, upfeat, get_spixel_image, build_LABXY_feat, label2one_hot_torch, rgb2Lab_torch, update_spixl_map
from superpixel_fcn.models.Spixel_single_layer import SpixelNet

from skimage.segmentation import mark_boundaries

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SuperPixel')
parser.add_argument('--sample_rate', type=int, default=0.0025, help="sample rate for pixel interpolation")
parser.add_argument('--train_img_width', type=int, default=960, help="default train image width")
parser.add_argument('--train_img_height', type=int, default=240, help="default train image height")
parser.add_argument('--input_img_width', type=int, default=960, help="default train image width")
parser.add_argument('--input_img_height', type=int, default=240, help="default train image height")
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning Rate. Default=0.0002')
parser.add_argument('--downsize', type=float, default=20, help='grid cell size for superpixel training ')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight_decay')
parser.add_argument('--bias_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--beta', type=float, default=0.999, help='Adam momentum term. Default=0.999')
parser.add_argument('--pos_weight', type=float, default=1.0, help='weight of the pos term')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/train.csv", help='path to train_csv')
parser.add_argument('--val_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/val.csv", help='path to val_csv')
parser.add_argument('--path_to_save', type=str, default="epochs_SuperPixeFCN_color", help='path to save trained models')
parser.add_argument('--path_to_tensorboard_log', type=str, default="tensorBoardRuns/SuperPixelFCN-color-loss-batch-8-bottom-crop-240x960-default-epoch-100-lr-000005-ADAM-c-001-color-loss-pos-weight-1-03-06-2021", help='path to tensorboard logging')
parser.add_argument('--device_ids', type=list, default=[0], help='path to tensorboard logging')

opt = parser.parse_args()

print(opt)


opt.path_to_save = opt.path_to_save + '_' + str(datetime.datetime.now())
if not os.path.exists(opt.path_to_save):
    os.makedirs(opt.path_to_save)
    
text_file = open(os.path.join(opt.path_to_save, "tensorboard_log_path"), "w")
text_file.write(opt.path_to_tensorboard_log)
text_file.close()

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets...')
# Please set the path to training and validation data here
# Suggest to put the data in SSD to get better data IO speed

train_set = KITTI_Dataset(opt.train_data_csv_path, True)
val_set   = KITTI_Dataset(opt.val_data_csv_path,   False)

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True )
val_loader   = DataLoader(dataset=val_set,   num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

print('===> Building model...')
model = SpixelNet(batchNorm=True)
model = nn.DataParallel(model, device_ids=opt.device_ids) #multi-GPU

if torch.cuda.is_available():
    model = model.cuda()
    
model.module.train()
print(model)

print('===> Parameters:', sum(param.numel() for param in model.parameters()))

print('===> Initialize Optimizer...')
param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': opt.bias_decay},
                {'params': model.module.weight_parameters(), 'weight_decay': opt.weight_decay}]

optimizer = optim.Adam(param_groups, lr=opt.lr, betas=(opt.momentum, opt.beta))
    
print('===> Initialize Logger...')     
configure(opt.path_to_tensorboard_log)

train_spixelID, train_XY_feat_stack = init_spixel_grid(opt)
val_spixelID,  val_XY_feat_stack = init_spixel_grid(opt, b_train=False)


def train(epoch):
    epoch_total_loss = 0.0
    epoch_sem_loss = 0.0
    epoch_pos_loss = 0.0
    
    epoch_start = time.time()
    
    model.module.train()
    
    #   Step up learning rate decay
    lr = opt.lr * (0.5 ** (epoch // (opt.nEpochs // 2)))
    
    optimizer = optim.Adam(param_groups, lr=lr, betas=(opt.momentum, opt.beta))

    for iteration, batch in enumerate(train_loader, 1):
        image_rgb, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        image_lab = rgb2Lab_torch(image_rgb.cuda()) # image in lab color space
        LABXY_feat_tensor = build_LABXY_feat(image_lab, train_XY_feat_stack)  # B* (3+2 )* H * W
                        
        if torch.cuda.is_available():
            image_rgb = image_rgb.cuda()
            LABXY_feat_tensor = LABXY_feat_tensor.cuda()
            
        torch.cuda.synchronize()

        #   Compute prediction
        optimizer.zero_grad()
        
        output = model(image_rgb) # white output

        slic_loss, loss_sem, loss_pos = compute_color_pos_loss(output, LABXY_feat_tensor, pos_weight= opt.pos_weight, kernel_size=opt.downsize)
        
        optimizer.zero_grad()
        slic_loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        
        epoch_total_loss += slic_loss.data.item()
        epoch_sem_loss += loss_sem.data.item()
        epoch_pos_loss += loss_pos.data.item()

    epoch_end = time.time()
    print("===> Epoch {} Complete: lr: {}, Avg. total_loss: {:.4f}, Avg. sem_loss: {:.4f}, Avg. epoch_pos_loss: {:.4f}, Time: {:.4f}".format(epoch, lr, epoch_total_loss/len(train_loader), epoch_sem_loss/len(train_loader), epoch_pos_loss/len(train_loader), (epoch_end-epoch_start)))
    
    log_value('epoch_total_loss', epoch_total_loss / len(train_loader), epoch)
    log_value('epoch_sem_loss', epoch_sem_loss / len(train_loader), epoch)
    log_value('epoch_pos_loss', epoch_pos_loss / len(train_loader), epoch)
    

def reshape_4D_array(array_4D, width_num):
    num, cha, height, width = array_4D.shape
    height_num = num // width_num
    total_width = width * width_num
    total_height = height * height_num
    target_array_4D = np.zeros((1, cha, total_height, total_width))
    for index in range(0, num):
        height_start = index//width_num
        width_start = index%width_num
        target_array_4D[:,:,height_start*height:height_start*height+height,width_start*width:width_start*width+width] = array_4D[index,:,:,:]
    
    if cha == 1:
        target_array_4D = np.repeat(target_array_4D, 3, axis=1)
        
                
    return target_array_4D

LOSS_best = math.inf

def val(epoch):
    avg_total_loss = 0.0
    avg_sem_loss = 0.0
    avg_pos_loss = 0.0
    
    frame_count = 0
    
    epoch_start = time.time()
    
    model.module.eval()
    
    for batch in val_loader:
        image_rgb, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        image_lab = rgb2Lab_torch(image_rgb.cuda()) # image in lab color space
        LABXY_feat_tensor = build_LABXY_feat(image_lab, train_XY_feat_stack)  # B* (3+2)
        
        if torch.cuda.is_available():
            image_rgb = image_rgb.cuda()
            LABXY_feat_tensor = LABXY_feat_tensor.cuda()
        
        torch.cuda.synchronize()
        
        with torch.no_grad():
            output = model(image_rgb) # white output
            
        torch.cuda.synchronize()
        
        slic_loss, loss_sem, loss_pos = compute_color_pos_loss(output, LABXY_feat_tensor,
                                                               pos_weight=opt.pos_weight, kernel_size=opt.downsize)
        
        avg_total_loss += slic_loss.data.item()
        avg_sem_loss += loss_sem.data.item()
        avg_pos_loss += loss_pos.data.item()

    
    epoch_end = time.time()
    print("===> Epoch {} Validation: Avg. total_loss: {:.4f}, Avg. sem_loss: {:.4f}, Avg. epoch_pos_loss: {:.4f}, Time: {:.4f}".format(epoch, avg_total_loss/len(val_loader), avg_sem_loss/len(val_loader), avg_pos_loss/len(val_loader), (epoch_end-epoch_start)))
    
    log_value('val_total_loss', avg_total_loss / len(val_loader), epoch)
    log_value('val_sem_loss', avg_sem_loss / len(val_loader), epoch)
    log_value('val_pos_loss', avg_pos_loss / len(val_loader), epoch)
    
    #   Draw the last image result
    spixl_map = update_spixl_map(val_spixelID[[-1],:,:,:], output[[-1],:,:,:]) # 1x1x240x960
    ori_sz_spixel_map =  F.interpolate(spixl_map.type(torch.float), size=(opt.input_img_height, opt.input_img_width), mode='nearest').type(torch.int)  # 1x1x240x960
    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1) #240x960
    spix_index_np = spix_index_np.astype(np.int64) #240x960, 1% here
    image_rgb_np = image_rgb[[-1],:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    spixel_bd_image = mark_boundaries(image_rgb_np, spix_index_np.astype(int), color=(0, 1, 1))
    spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)
    spixel_viz = np.expand_dims(spixel_viz, axis=0)
    image_rgb_np_viz = image_rgb_np.astype(np.float32).transpose(2, 0, 1)
    image_rgb_np_viz = np.expand_dims(image_rgb_np_viz, axis=0)
    
    log_images('spixel', reshape_4D_array(spixel_viz, 1), step=1)
    log_images('image_rgb', reshape_4D_array(image_rgb_np_viz, 1), step=1)
        
    global LOSS_best
    if avg_total_loss < LOSS_best:
        LOSS_best = avg_total_loss
        model_out_path = opt.path_to_save + "/model_best.pth".format(epoch)
        torch.save(model.module.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))


def checkpoint(epoch):
    if epoch%1 == 0:
        if not os.path.exists(opt.path_to_save):
            os.makedirs(opt.path_to_save)
        model_out_path = opt.path_to_save + "/model_epoch_{}.pth".format(epoch)
        torch.save(model.module.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

val(0)
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val(epoch)
    checkpoint(epoch)
