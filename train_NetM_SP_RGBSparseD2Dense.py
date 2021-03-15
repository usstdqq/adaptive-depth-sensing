from __future__ import print_function

import os
import sys
THIS_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_PATH, "superpixel_fcn"))

import csv
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import datetime
import numpy as np
import math

from math import log10
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from data_utils import NYU_V2_Dataset
from model import NetME_RGBSparseD2Dense
from tensorboard_logger import configure, log_value, log_images
from loss import MaskedMSELoss, MaskedL1Loss
from metrics import AverageMeter, Result
from dataloaders.data_utils import KITTI_Dataset

from superpixel_fcn.train_util import init_spixel_grid, rgb2Lab_torch, update_spixl_map, build_LABXY_feat
from superpixel_fcn.loss import compute_pos_loss, compute_color_pos_loss

from skimage.segmentation import mark_boundaries


# Training settings
parser = argparse.ArgumentParser(description='PyTorch NetRGBM-NetE')
parser.add_argument('--sample_rate', type=int, default=0.0025, help="sample rate for pixel interpolation")
parser.add_argument('--downsize', type=float, default=20, help='grid cell size for superpixel training')
parser.add_argument('--img_width', type=int, default=960, help="default train image width")
parser.add_argument('--img_height', type=int, default=240, help="default train image height")
parser.add_argument('--train_img_width', type=int, default=960, help="default train image width")
parser.add_argument('--train_img_height', type=int, default=240, help="default train image height")
parser.add_argument('--input_img_width', type=int, default=960, help="default train image width")
parser.add_argument('--input_img_height', type=int, default=240, help="default train image height")
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--temperature', type=float, default=1.0, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0002')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--beta', type=float, default=0.999, help='Adam momentum term. Default=0.999')
parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight_decay')
parser.add_argument('--bias_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--pos_weight', type=float, default=1, help='inside slic loss')
parser.add_argument('--slic_weight', type=float, default=0.000001, help='to enforce regular shape of super pixel')
parser.add_argument('--proj_weight', type=float, default=0.0, help='to enforce regular shape of super pixel')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/train.csv", help='path to train_csv')
parser.add_argument('--val_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/val.csv", help='path to val_csv')
parser.add_argument('--path_to_save', type=str, default="epochs_NetM_RGBSparseD_wd", help='path to save trained models')
parser.add_argument('--path_to_tensorboard_log', type=str, default="tensorBoardRuns/NetM-SP-RGBSparseD-linear-bilinear-clip-batch-8-240x960-crop-default-epoch-100-lr-00001-decay-slic-w-0000001-pos-w-1-proj-w-0-anneal-temp-1-01-ADAM-SGD-c-00025-L1-loss-03-14-2021", help='path to tensorboard logging')
parser.add_argument('--path_to_NetE_pre', type=str, default="epochs_S2D_SparseD_wd_2021-03-07 22:44:42.182760/model_epoch_100.pth", help='path to tensorboard logging')
parser.add_argument('--path_to_NetSP_pre', type=str, default="epochs_SuperPixeFCN_color_2021-03-06 20:47:57.179621/model_epoch_100.pth", help='path to tensorboard logging')
parser.add_argument('--device_ids', type=list, default=[0, 1], help='path to tensorboard logging')
parser.add_argument('--nef', type=int, default=16, help='number of encoder filters in first conv layer')


opt = parser.parse_args()

print(opt)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
              'delta1', 'delta2', 'delta3',
              'data_time', 'gpu_time']

best_result = Result()
best_result.set_to_worst()

opt.path_to_save = opt.path_to_save + '_' + str(datetime.datetime.now())
if not os.path.exists(opt.path_to_save):
    os.makedirs(opt.path_to_save)

text_file = open(os.path.join(opt.path_to_save, "tensorboard_log_path"), "w")
text_file.write(opt.path_to_tensorboard_log)
text_file.close()
    
train_csv = os.path.join(opt.path_to_save, 'train.csv')
val_csv = os.path.join(opt.path_to_save, 'val.csv')
val_rand_csv = os.path.join(opt.path_to_save, 'val_rand.csv')

# create new csv files with only header
with open(train_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
with open(val_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
with open(val_rand_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


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
modelME = NetME_RGBSparseD2Dense(opt.path_to_NetE_pre, opt.path_to_NetSP_pre, opt.sample_rate, opt.img_height, opt.img_width, opt.downsize, opt.batch_size, opt.temperature)
modelME = nn.DataParallel(modelME, device_ids=opt.device_ids) #multi-GPU
criterion_mse = MaskedMSELoss()
criterion_depth = MaskedL1Loss()

if torch.cuda.is_available():
    modelME = modelME.cuda()    
    criterion_mse = criterion_mse.cuda()
    criterion_depth = criterion_depth.cuda()
    
modelME.module.netM.train()
modelME.module.netE.eval()
print(modelME)

print('===> Parameters:', sum(param.numel() for param in modelME.parameters()))

print('===> Initialize Optimizer...')      
optimizer = optim.Adam([{'params': modelME.module.netM.bias_parameters(), 'lr': opt.lr, 'weight_decay': opt.bias_decay},
                        {'params': modelME.module.netM.weight_parameters(), 'lr': opt.lr, 'weight_decay': opt.weight_decay},
                        {'params': modelME.module.netE.bias_parameters(), 'lr': 0.0, 'weight_decay': opt.bias_decay},
                        {'params': modelME.module.netE.weight_parameters(), 'lr': 0.0, 'weight_decay': opt.weight_decay}
                        ], lr=opt.lr, betas=(opt.momentum, opt.beta))

# optimizer = optim.SGD([{'params': modelME.module.netM.bias_parameters(), 'lr': opt.lr, 'weight_decay': opt.bias_decay},
#                        {'params': modelME.module.netM.weight_parameters(), 'lr': opt.lr, 'weight_decay': opt.weight_decay},
#                        {'params': modelME.module.netE.bias_parameters(), 'lr': 0.0, 'weight_decay': opt.bias_decay},
#                        {'params': modelME.module.netE.weight_parameters(), 'lr': 0.0, 'weight_decay': opt.weight_decay}
#                        ], lr=opt.lr, momentum=opt.momentum)


print('===> Initialize Logger...')     
configure(opt.path_to_tensorboard_log)

train_spixelID, train_XY_feat_stack = init_spixel_grid(opt)
val_spixelID,  val_XY_feat_stack = init_spixel_grid(opt, b_train=False)

def train(epoch):
    epoch_loss = 0
    epoch_loss_depth = 0
    epoch_loss_pos = 0
    epoch_loss_temperature = 0
    epoch_psnr = 0
    epoch_sparsity = 0
    
    epoch_loss_slic = 0.0
    epoch_loss_color = 0.0
    epoch_loss_pos = 0.0
    
    epoch_start = time.time()
    end = time.time()
    
    average_meter = AverageMeter()
    
    # train/eval modes make difference on batch normalization layer
    modelME.module.netM.train()
    modelME.module.netE.eval()
    
    # setup learning rate decay
    lr = opt.lr * (0.5 ** (epoch // (opt.nEpochs // 5)))
    
    # setup temperature for SSA
    temperature = opt.temperature * (1 - 0.9 * epoch / opt.nEpochs)
    
    modelME.module.netM.temperature.fill_(temperature)
    
    # use ADAM for the first 2 epoch, then SGD, to speedup training
    if epoch <=2:
        optimizer = optim.Adam([{'params': modelME.module.netM.bias_parameters(), 'lr': opt.lr, 'weight_decay': opt.bias_decay},
                                {'params': modelME.module.netM.weight_parameters(), 'lr': opt.lr, 'weight_decay': opt.weight_decay},
                                {'params': modelME.module.netE.bias_parameters(), 'lr': 0.0, 'weight_decay': opt.bias_decay},
                                {'params': modelME.module.netE.weight_parameters(), 'lr': 0.0, 'weight_decay': opt.weight_decay}
                                ], lr=lr, betas=(opt.momentum, opt.beta))
    else:
        optimizer = optim.SGD([{'params': modelME.module.netM.bias_parameters(), 'lr': opt.lr, 'weight_decay': opt.bias_decay},
                               {'params': modelME.module.netM.weight_parameters(), 'lr': opt.lr, 'weight_decay': opt.weight_decay},
                               {'params': modelME.module.netE.bias_parameters(), 'lr': 0.0, 'weight_decay': opt.bias_decay},
                               {'params': modelME.module.netE.weight_parameters(), 'lr': 0.0, 'weight_decay': opt.weight_decay}
                               ], lr=lr, momentum=opt.momentum)

    for iteration, batch in enumerate(train_loader, 1):
        image_target, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        image_target_lab = rgb2Lab_torch(image_target.cuda()) # image in lab color space
        image_target_labxy_feat_tensor = build_LABXY_feat(image_target_lab, train_XY_feat_stack)  # B* (3+2 )* H * W
        
        depth_input = depth_target.clone()
        
        if torch.cuda.is_available():
            image_target = image_target.cuda()
            depth_target = depth_target.cuda()
            depth_input = depth_input.cuda()
            depth_mask = depth_mask.cuda()
            image_target_labxy_feat_tensor = image_target_labxy_feat_tensor.cuda()
        
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        #   Compute prediction
        end = time.time()
        optimizer.zero_grad()
        
        depth_recon, corrupt_mask_soft, corrupt_mask_binary, pooled_xy_tensor, reconstr_xy_tensor, curr_spixl_map, prob = modelME(image_target, depth_input, True)
    
        mask_sparsity = corrupt_mask_binary.sum() / (corrupt_mask_binary.shape[0] * corrupt_mask_binary.shape[1] * corrupt_mask_binary.shape[2] * corrupt_mask_binary.shape[3])

        batch_size_cur = image_target.shape[0]
        loss_depth = criterion_depth(depth_recon, depth_target, depth_mask)
        loss_mse = criterion_mse(depth_recon, depth_target, depth_mask) # in [0, 1 range]
        loss_slic, loss_color, loss_pos = compute_color_pos_loss(prob, image_target_labxy_feat_tensor[:batch_size_cur,:,:,:], pos_weight= opt.pos_weight, kernel_size=opt.downsize)

        loss_temperature = modelME.module.netM.temperature ** 2
                
        loss = loss_depth + opt.slic_weight * loss_slic + opt.proj_weight * loss_temperature
        loss.backward()
        
        optimizer.step()
        
        torch.cuda.synchronize()
                
        psnr = 10 * log10(1 / loss_mse.data.item())
        epoch_loss += loss.data.item()
        epoch_loss_depth += loss_depth.data.item()
        epoch_loss_pos += loss_pos.data.item()
        epoch_loss_temperature += loss_temperature.data.item()
        epoch_psnr += psnr
        epoch_sparsity += mask_sparsity
        
        epoch_loss_slic += loss_slic.data.item()
        epoch_loss_color += loss_color.data.item()
        epoch_loss_pos += loss_pos.data.item()
        
        gpu_time = time.time() - end
                
        # measure accuracy and record loss
        result = Result()
        result.evaluate(depth_recon.data, depth_target.data)
        average_meter.update(result, gpu_time, data_time, image_target.size(0))

    epoch_end = time.time()
    print("===> Epoch {} Complete: lr: {}, Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, lr, epoch_loss / len(train_loader), epoch_psnr / len(train_loader), (epoch_end-epoch_start)))
    
    log_value('train_loss', epoch_loss / len(train_loader), epoch)
    log_value('train_loss_depth', epoch_loss_depth / len(train_loader), epoch)
    log_value('train_loss_pos', epoch_loss_pos / len(train_loader), epoch)
    log_value('train_loss_temperature', epoch_loss_temperature / len(train_loader), epoch)
    log_value('train_psnr', epoch_psnr / len(train_loader), epoch)
    log_value('train_sparsity', epoch_sparsity / len(train_loader), epoch) 
    
    log_value('train_loss_slic', epoch_loss_slic / len(train_loader), epoch)
    log_value('train_loss_color', epoch_loss_color / len(train_loader), epoch)
    log_value('train_loss_pos', epoch_loss_pos / len(train_loader), epoch)

    print('Train Epoch: {0} [{1}/{2}]\t'
              't_Data={data_time:.3f}({average.data_time:.3f}) '
              't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
              'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
              'MAE={result.mae:.2f}({average.mae:.2f}) '
              'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
              'REL={result.absrel:.3f}({average.absrel:.3f}) '
              'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
              epoch, 1, len(train_loader), data_time=data_time,
              gpu_time=gpu_time, result=result, average=average_meter.average()))
    
    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

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
        target_array_4D = target_array_4D / 10.0
        target_array_4D = np.repeat(target_array_4D, 3, axis=1)
                
    return target_array_4D

LOSS_best = math.inf

def val(epoch):
    avg_loss = 0
    avg_loss_depth = 0
    avg_loss_pos = 0
    avg_loss_temperature = 0
    avg_psnr = 0
    avg_sparsity = 0
    
    avg_loss_slic = 0.0
    avg_loss_color = 0.0
    avg_loss_pos = 0.0
    
    frame_count = 0
    
    epoch_start = time.time()
    end = time.time()
    
    average_meter = AverageMeter()
    
    modelME.module.eval()
    modelME.module.netM.eval()
    modelME.module.netE.eval()
    
    for batch in val_loader:
        image_target, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        image_target_lab = rgb2Lab_torch(image_target.cuda()) # image in lab color space
        image_target_labxy_feat_tensor = build_LABXY_feat(image_target_lab, train_XY_feat_stack)  # B* (3+2 )* H * W
        
        depth_input = depth_target.clone()

        if torch.cuda.is_available():
            image_target = image_target.cuda()
            depth_input = depth_input.cuda()
            depth_target = depth_target.cuda()
            depth_mask = depth_mask.cuda()
            image_target_labxy_feat_tensor = image_target_labxy_feat_tensor.cuda()
        
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()
        with torch.no_grad():
            depth_recon, corrupt_mask_soft, corrupt_mask_binary, pooled_xy_tensor, reconstr_xy_tensor, curr_spixl_map, prob = modelME(image_target, depth_input, False)
        
        torch.cuda.synchronize()

        mask_sparsity = corrupt_mask_binary.sum() / (corrupt_mask_binary.shape[0] * corrupt_mask_binary.shape[1] * corrupt_mask_binary.shape[2] * corrupt_mask_binary.shape[3])
        
         #   Generate the corrupted depth image
        depth_input = corrupt_mask_binary * depth_input
        
        rgb_sparse_d_input = torch.cat((image_target, depth_input), 1) # white input
        
        with torch.no_grad():
            restored_depth = modelME.module.netE(rgb_sparse_d_input)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        # measure accuracy and record loss
        result = Result()
        result.evaluate(restored_depth.data, depth_target.data)
        
        average_meter.update(result, gpu_time, data_time, depth_input.size(0))
        end = time.time()
        

        for i in range(0, depth_input.shape[0]):
            
            loss_depth = criterion_depth(restored_depth[[i]], depth_target[[i]], depth_mask[[i]])
            loss_mse = criterion_mse(restored_depth[[i]], depth_target[[i]], depth_mask[[i]])
            psnr = 10 * log10(1 / loss_mse.data.item())
                    
            loss_slic, loss_color, loss_pos = compute_color_pos_loss(prob[[i]], image_target_labxy_feat_tensor[[i]], pos_weight=opt.pos_weight, kernel_size=opt.downsize)
        
            loss_temperature = modelME.module.netM.temperature ** 2
            
            loss = loss_depth + opt.slic_weight * loss_slic + opt.proj_weight * loss_temperature
            
            avg_loss += loss.data.item()
            avg_loss_depth += loss_depth.data.item()
            avg_loss_pos += loss_pos.data.item()
            avg_loss_temperature += loss_temperature.data.item()
            avg_psnr += psnr
            avg_sparsity += mask_sparsity
            
            avg_loss_slic += loss_slic.data.item()
            avg_loss_color += loss_color.data.item()
            avg_loss_pos += loss_pos.data.item()
            
            frame_count += 1
            
    avg = average_meter.average()
    
    print('\n*\n'
    'RMSE={average.rmse:.3f}\n'
    'MAE={average.mae:.3f}\n'
    'Delta1={average.delta1:.3f}\n'
    'REL={average.absrel:.3f}\n'
    'Lg10={average.lg10:.3f}\n'
    't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))

    with open(val_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
            
    epoch_end = time.time()
    print("===> Epoch {} Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Mask Sparsity: {:.4f}, Time: {:.4f}".format(epoch, avg_loss / frame_count, avg_psnr / frame_count, avg_sparsity / frame_count,  (epoch_end-epoch_start)))

    log_value('val_loss', avg_loss / frame_count, epoch)
    log_value('val_loss_depth', avg_loss_depth / frame_count, epoch)
    log_value('val_loss_pos', avg_loss_pos / frame_count, epoch)
    log_value('val_loss_temperature', avg_loss_temperature / frame_count, epoch)
    log_value('val_psnr', avg_psnr / frame_count, epoch)
    log_value('val_sparsity', avg_sparsity / frame_count, epoch) 
    
    log_value('val_loss_slic', avg_loss_slic / frame_count, epoch)
    log_value('val_loss_color', avg_loss_color / frame_count, epoch)
    log_value('val_loss_pos', avg_loss_pos / frame_count, epoch)
   
    #   Draw the last image result
    spixl_map = update_spixl_map(val_spixelID[[-1],:,:,:], prob[[-1],:,:,:]) # 1x1x240x960
    ori_sz_spixel_map =  F.interpolate(spixl_map.type(torch.float), size=(opt.input_img_height, opt.input_img_width), mode='nearest').type(torch.int)  # 1x1x240x960
    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1) #240x960
    spix_index_np = spix_index_np.astype(np.int64) #240x960, 1% here
    image_rgb_np = image_target[[-1],:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    spixel_bd_image = mark_boundaries(image_rgb_np, spix_index_np.astype(int), color=(0, 1, 1))
    spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)
    spixel_viz = np.expand_dims(spixel_viz, axis=0)
    
    log_images('image_target_spixel', reshape_4D_array(spixel_viz, 1), step=1)
    
    log_images('depth_input', reshape_4D_array(depth_input[[-1],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_prediction', reshape_4D_array(restored_depth[[-1],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_target', reshape_4D_array(depth_target[[-1],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('image_target', reshape_4D_array(image_target[[-1],:,:,:].data.cpu().numpy(), 1), step=1)   
    log_images('corrupt_mask_binary', reshape_4D_array(corrupt_mask_binary[[-1],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('corrupt_mask_soft', reshape_4D_array(corrupt_mask_soft[[-1],:,:,:].data.cpu().numpy(), 1), step=1)
        

    global LOSS_best
    if avg_loss_depth < LOSS_best:
        LOSS_best = avg_loss
        model_out_path = opt.path_to_save + "/model_best.pth"
        torch.save(modelME.module.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))


def checkpoint(epoch):
    if epoch%10 == 0:
        if not os.path.exists(opt.path_to_save):
            os.makedirs(opt.path_to_save)
        model_out_path = opt.path_to_save + "/model_epoch_{}.pth".format(epoch)
        torch.save(modelME.module.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        
def val_rand(epoch):
    avg_loss = 0
    avg_psnr = 0
    frame_count = 0
    
    epoch_start = time.time()
    end = time.time()
    
    average_meter = AverageMeter()
        
    modelME.module.eval()
    modelME.module.netM.eval()
    modelME.module.netE.eval()
    
    for batch in val_loader:
        image_target, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        batch_size = depth_target.size(0)
        depth_input = depth_target.clone()
        
        image_height = image_target.size(2)
        image_width  = image_target.size(3)
        #   Corrupt the target image
        for i in range(0, batch_size):
            n_depth_mask = depth_mask[i,0,:,:].sum().item()
            
            #   Adjust the sampling rate based on the depth_mask
            sample_rate = opt.sample_rate / n_depth_mask * (image_height * image_width)
            
            corrupt_mask = np.random.binomial(1, (1 - sample_rate), (image_height, image_width))
            
            corrupt_mask.astype(np.bool)
            corrupt_mask = torch.BoolTensor(corrupt_mask)
            
            depth_input[i,0,:,:].masked_fill_(corrupt_mask, (0.0))
            
        if torch.cuda.is_available():
            image_target = image_target.cuda()
            depth_input = depth_input.cuda()
            depth_target = depth_target.cuda()
            depth_mask = depth_mask.cuda()
            
        rgb_sparse_d_input = torch.cat((image_target, depth_input), 1)
            
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            depth_prediction = modelME.module.netE(rgb_sparse_d_input)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        # measure accuracy and record loss
        result = Result()
        result.evaluate(depth_prediction.data, depth_target.data)
        average_meter.update(result, gpu_time, data_time, depth_input.size(0))
        end = time.time()
        
        for i in range(0, depth_input.shape[0]):

            loss_depth = criterion_depth(depth_prediction[[i]], depth_target[[i]], depth_mask[[i]])
            loss_mse = criterion_mse(depth_prediction[[i]], depth_target[[i]], depth_mask[[i]])
            psnr = 10 * log10(1 / loss_mse.data.item())
            
            avg_loss  += loss_depth.data.item()
            avg_psnr  += psnr
            frame_count += 1
            
    avg = average_meter.average()
    
    print('\n*\n'
    'RMSE={average.rmse:.3f}\n'
    'MAE={average.mae:.3f}\n'
    'Delta1={average.delta1:.3f}\n'
    'REL={average.absrel:.3f}\n'
    'Lg10={average.lg10:.3f}\n'
    't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))
    
    with open(val_rand_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    
    epoch_end = time.time()
    print("===> Epoch {} Random Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, avg_loss / frame_count, avg_psnr / frame_count, (epoch_end-epoch_start)))
    
    log_value('val_loss_depth_rand', avg_loss / frame_count, epoch)
    log_value('val_psnr_rand', avg_psnr / frame_count, epoch)
    
    log_images('depth_input_rand', reshape_4D_array(depth_input[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_prediction_rand', reshape_4D_array(depth_prediction[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_target_rand', reshape_4D_array(depth_target[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('image_target_rand', reshape_4D_array(image_target[[0],:,:,:].data.cpu().numpy(), 1), step=1)  

val(0)
val_rand(0)
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val(epoch)
    val_rand(epoch)
    checkpoint(epoch)
