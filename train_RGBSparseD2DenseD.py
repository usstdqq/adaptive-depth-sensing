from __future__ import print_function
import os
import csv
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import numpy as np
import math

from math import log10
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fusion_models.model import uncertainty_net
from tensorboard_logger import configure, log_value, log_images
from loss import MaskedMSELoss, MaskedL1Loss
from metrics import AverageMeter, Result
from dataloaders.data_utils_nyuv2 import NYUDataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NetE')
parser.add_argument('--sample_rate', type=int, default=0.0025, help="sample rate for pixel interpolation")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.0002')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay for SGD')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam momentum term. Default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_data_path', type=str, default="/home/dqq/Data/nyudepthv2_h5/train", help='path to train_csv')
parser.add_argument('--val_data_path', type=str, default="/home/dqq/Data/nyudepthv2_h5/val", help='path to val_csv')
parser.add_argument('--path_to_save', type=str, default="epochs_S2D_RGBSparseD_wd", help='path to save trained models')
parser.add_argument('--path_to_tensorboard_log', type=str, default="tensorBoardRuns/S2D-RGBSparseD-linear-bilinear-clip-batch-16-240x320-crop-default-nyusize-epoch-100-lr-0001-decay-ADAM-c-00025-L2-loss-03-27-2021", help='path to tensorboard logging')
parser.add_argument('--device_ids', type=list, default=[0, 1], help='path to tensorboard logging')
parser.add_argument('--wlid', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wrgb', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wpred', type=float, default=1, help="weight base loss")
parser.add_argument('--wguide', type=float, default=0.1, help="weight base loss")
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

# create new csv files with only header
with open(train_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
with open(val_csv, 'w') as csvfile:
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

train_set = NYUDataset(opt.train_data_path, type='train', modality='rgb', sparsifier=None)
val_set   = NYUDataset(opt.val_data_path,   type='val',   modality='rgb', sparsifier=None)

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True )
val_loader   = DataLoader(dataset=val_set,   num_workers=opt.threads, batch_size=4, shuffle=False)

print('===> Building model...')
model = uncertainty_net(in_channels=4, thres=0)
model = nn.DataParallel(model, device_ids=opt.device_ids) #multi-GPU
criterion_mse = MaskedMSELoss()
criterion_depth = MaskedMSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion_mse = criterion_mse.cuda()
    criterion_depth = criterion_depth.cuda()
    
model.module.train()
print(model)

print('===> Parameters:', sum(param.numel() for param in model.parameters()))

print('===> Initialize Optimizer...')

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
print('===> Initialize Logger...')     
configure(opt.path_to_tensorboard_log)


def train(epoch):
    epoch_loss = 0
    epoch_psnr = 0
    
    epoch_start = time.time()
    end = time.time()
    
    average_meter = AverageMeter()
    
    model.module.train()
    
    #   Step up learning rate decay
    lr = opt.lr * (0.2 ** (epoch // (opt.nEpochs // 4)))
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    # lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, 0.999))

    for iteration, batch in enumerate(train_loader, 1):
        image_target, depth_target, depth_mask = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        batch_size = depth_target.size(0)
        depth_input = depth_target.clone()
    
        image_height = image_target.size(2)
        image_width  = image_target.size(3)
        
        #   Corrupt the depth_input image
        for i in range(0, batch_size):
            n_depth_mask = depth_mask[i,0,:,:].sum().item()
            
            #   Adjust the sampling rate based on the depth_mask
            if n_depth_mask != 0:
                sample_rate = opt.sample_rate / n_depth_mask * (image_height * image_width)   
                sample_rate = np.clip(sample_rate, 0.0, 1.0) # some randome augementation can cause some crazy mask
            else:
                sample_rate = opt.sample_rate
                
            corrupt_mask = np.random.binomial(1, (1 - sample_rate), (image_height, image_width))
            corrupt_mask.astype(np.bool)
            corrupt_mask = torch.BoolTensor(corrupt_mask)
            
            depth_input[i,0,:,:].masked_fill_(corrupt_mask, (0.0))
        
        if torch.cuda.is_available():
            depth_input = depth_input.cuda()
            depth_target = depth_target.cuda()
            image_target = image_target.cuda()
            depth_mask = depth_mask.cuda()
            
        torch.cuda.synchronize()
        data_time = time.time() - end

        #   Compute prediction
        end = time.time()
        optimizer.zero_grad()
        
        rgb_sparse_d_input = torch.cat((depth_input, image_target), 1) # white input
        
        depth_prediction, lidar_out, precise, guide = model(rgb_sparse_d_input) # white output
        
        loss_depth = criterion_depth(depth_prediction, depth_target, depth_mask)
        loss_lidar = criterion_depth(lidar_out[:,[0],:,:], depth_target, depth_mask)
        loss_rgb = criterion_depth(precise, depth_target, depth_mask)
        loss_guide = criterion_depth(guide, depth_target, depth_mask)
            
        loss_depth = opt.wpred*loss_depth + opt.wlid*loss_lidar + opt.wrgb*loss_rgb + opt.wguide*loss_guide
        
        loss_mse = criterion_mse(depth_prediction, depth_target, depth_mask)
        
        psnr = 10 * log10(1 / loss_mse.data.item())
        epoch_loss += loss_depth.data.item()
        epoch_psnr += psnr
        loss_depth.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        # measure accuracy and record loss
        result = Result()
        result.evaluate(depth_prediction.data, depth_target.data)
        average_meter.update(result, gpu_time, data_time, image_target.size(0))
        
        end = time.time()

    epoch_end = time.time()
    print("===> Epoch {} Complete: lr: {}, Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, lr, epoch_loss / len(train_loader), epoch_psnr / len(train_loader), (epoch_end-epoch_start)))
    
    log_value('train_loss', epoch_loss / len(train_loader), epoch)
    log_value('train_psnr', epoch_psnr / len(train_loader), epoch)
    
    print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
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
        target_array_4D = target_array_4D / 100
        target_array_4D = np.repeat(target_array_4D, 3, axis=1)
        
                
    return target_array_4D

LOSS_best = math.inf

def val(epoch):
    avg_loss = 0
    avg_psnr = 0
    frame_count = 0
    
    epoch_start = time.time()
    end = time.time()
    
    average_meter = AverageMeter()
    model.module.eval()
    
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
            depth_input = depth_input.cuda()
            depth_target = depth_target.cuda()
            image_target = image_target.cuda()
            depth_mask = depth_mask.cuda()

        rgb_sparse_d_input = torch.cat((depth_input, image_target), 1) # white input
        
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()
        with torch.no_grad():
            depth_prediction, lidar_out, precise, guide = model(rgb_sparse_d_input) # white output
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        # measure accuracy and record loss
        result = Result()
        # result.evaluate(depth_prediction.data * DEPTH_STD + DEPTH_MEAN, depth_target.data * DEPTH_STD + DEPTH_MEAN)
        result.evaluate(depth_prediction.data, depth_target.data)
        average_meter.update(result, gpu_time, data_time, depth_input.size(0))
        end = time.time()
        
        for i in range(0, depth_input.shape[0]):
                        
            loss_depth = criterion_depth(depth_prediction[[i]], depth_target[[i]], depth_mask[[i]])

            loss_mse = criterion_mse(depth_prediction[[i]], depth_target[[i]], depth_mask[[i]])
            psnr = 10 * log10(1 / loss_mse.data.item())
            
            avg_loss += loss_depth.data.item()
            avg_psnr += psnr
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
    print("===> Epoch {} Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, avg_loss / frame_count, avg_psnr / frame_count, (epoch_end-epoch_start)))
    
    log_value('val_loss', avg_loss / frame_count, epoch)
    log_value('val_psnr', avg_psnr / frame_count, epoch)
    
    log_images('depth_input', reshape_4D_array(depth_input[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_prediction', reshape_4D_array(depth_prediction[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('depth_target', reshape_4D_array(depth_target[[0],:,:,:].data.cpu().numpy(), 1), step=1)
    log_images('image_target', reshape_4D_array(image_target[[0],:,:,:].data.cpu().numpy(), 1), step=1)   
    
    global LOSS_best
    if avg_loss < LOSS_best:
        LOSS_best = avg_loss
        model_out_path = opt.path_to_save + "/model_best.pth".format(epoch)
        torch.save(model.module.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))


def checkpoint(epoch):
    if epoch%10 == 0:
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
