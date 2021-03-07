from __future__ import absolute_import

import os
import sys
THIS_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_PATH, "superpixel_fcn"))

import argparse
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import torchvision
import math
import collections
import torchvision.models

from superpixel_fcn.models.Spixel_single_layer import SpixelNet
from superpixel_fcn.train_util import init_spixel_grid, poolfeat, upfeat, get_spixel_image, build_LABXY_feat, label2one_hot_torch, rgb2Lab_torch, init_IDX_XY_grid, update_spixl_map

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time

RGB_MEAN = 0.5
RGB_STD = 0.5

DEPTH_MIN = 0.0
DEPTH_MAX = 100.0

EPS = 0.00001

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
            # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)


    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        
        x = self.bilinear(x)
        
        x = torch.clamp(x, min=DEPTH_MIN, max=DEPTH_MAX)
        
        return x
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    
    
# NetE ends here
# ====================================================================================
# NetM starts here     
class Mean_Shift(nn.Module):
    def __init__(self, sample_rate=0.2):
        super(Mean_Shift, self).__init__()
#        self.sample_rate = sample_rate
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
#        self.sample_rate = torch.autograd.Variable(torch.tensor(sample_rate), requires_grad=False)
        if torch.cuda.is_available(): self.sample_rate = self.sample_rate.cuda()
	
    def forward(self, x):
        x_size = x.size()
        
        x_mean = torch.mean(x, 2, True)
        x_mean = torch.mean(x_mean, 3, True)
        x_mean = x_mean.expand(x_size[0], x_size[1], x_size[2], x_size[3])
        
        x_out = x / x_mean * self.sample_rate
        
        return x_out
        


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(16, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(16, affine=True)
        
#        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#        self.in1 = nn.InstanceNorm2d(64, affine=True)
#        self.relu = nn.LeakyReLU(0.2, inplace=True)
#        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class BerLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
#        ctx.save_for_backward(input)
        return input.clamp(min=0.0, max=1.0).bernoulli()

    @staticmethod
    def backward(ctx, grad_output):
#        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

    
class NetM(nn.Module):
    def __init__(self, path_to_NetSP_pre, sample_rate, img_height, img_width, down_size, batch_size, temperature_init, kernel_size=7):
        super(NetM, self).__init__()
        
        self.sample_rate = sample_rate
        self.img_height = img_height
        self.img_width = img_width
        self.down_size = down_size
        self.batch_size = batch_size
        
        self.pooled_img_height = int(img_height/down_size)
        self.pooled_img_width = int(img_width/down_size)
        
        self.super_pixel_model = SpixelNet(batchNorm=True)
        self.super_pixel_model.load_state_dict(torch.load(path_to_NetSP_pre))
        
        # Bx9xHxW, Bx2xHxW
        self.spixel_id_tensor, self.xy_tensor = init_IDX_XY_grid(self.img_height, self.img_width, self.down_size, self.batch_size)
        self.spixel_id_tensor.requires_grad = False
        self.xy_tensor.requires_grad = False
        
        if torch.cuda.is_available():
            self.spixel_id_tensor = self.spixel_id_tensor.cuda()
            self.xy_tensor = self.xy_tensor.cuda()

    def warp_sample(self, pooled_xy_tensor, dense_depth_gt):
        
        grid_h = pooled_xy_tensor[:, [1], :, :].clone() # Bx1xhxw
        grid_w = pooled_xy_tensor[:, [0], :, :].clone() # Bx1xhxw
        
        grid_h = (2.0*grid_h - self.img_height) / self.img_height
        grid_w = (2.0*grid_w - self.img_width) / self.img_width
        
        
        grid_to_sample = torch.cat((grid_w, grid_h), dim=1) # Bx2xhxw
        # grid_to_sample = torch.cat((grid_h, grid_w), dim=1) # Bx2xhxw
        grid_to_sample = grid_to_sample.permute(0, 2, 3, 1)# Bxhxwx2

        # broadcast self.norm_grid over

        depth_sampled = F.grid_sample(dense_depth_gt, grid_to_sample, align_corners=True)# Bx1xhxw
        
        sparse_depth = self.expand(depth_sampled, pooled_xy_tensor)

        return sparse_depth, grid_to_sample
    
    def expand(self, depth_sampled, pooled_xy_tensor):
        sparse_depth = torch.zeros(depth_sampled.shape[0],
                                   depth_sampled.shape[1],
                                   self.img_height,
                                   self.img_width)
        
        if pooled_xy_tensor.is_cuda:
            sparse_depth = sparse_depth.cuda()
            
        for b in range(depth_sampled.shape[0]):
            grid_xp = pooled_xy_tensor[b,0,:,:] #24x96
            grid_yp = pooled_xy_tensor[b,1,:,:] #24x96
            
            row_list = grid_yp.view(-1) # 2304
            col_list = grid_xp.view(-1) # 2304
            
            row_list = torch.round(row_list)
            col_list = torch.round(col_list)
            
            row_list = torch.clamp(row_list, 0, self.img_height-1)
            col_list = torch.clamp(col_list, 0, self.img_width-1)
            
            row_list = row_list.long()
            col_list = col_list.long()
            
            row_col_list = torch.stack([row_list, col_list], dim=0)

            
            value_list = depth_sampled[b,:,:,:].view(-1) # 2304
            
            sparse_depth[b,0,:,:] = torch.sparse.FloatTensor(row_col_list, value_list, torch.Size([self.img_height,self.img_width])).to_dense()
            

        return sparse_depth    
        
    def forward(self, image_rgb, image_depth):
        
        input_batch_size = image_rgb.shape[0]
        
        spixel_id_tensor, xy_tensor = init_IDX_XY_grid(self.img_height, self.img_width, self.down_size, input_batch_size)
        
        prob = self.super_pixel_model(image_rgb) #  Bx9xHxW
        
        curr_spixl_map = update_spixl_map(spixel_id_tensor, prob) #  Bx1xHxW, super pixel index map
        
        pooled_xy_tensor = poolfeat(xy_tensor, prob, self.down_size, self.down_size) # Bx2xhxw
        
        sparse_depth, grid_to_sample = self.warp_sample(pooled_xy_tensor, image_depth)
        
        sampling_mask = sparse_depth > 0.0
    
        
        return sparse_depth, sampling_mask, pooled_xy_tensor, curr_spixl_map, prob

    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    
class NetME_RGBSparseD2Dense(nn.Module):
    def __init__(self, NetE_path, NetSP_path, sample_rate, img_height, img_width, down_size, batch_size, temperature_init):
        super(NetME_RGBSparseD2Dense, self).__init__()
        self.netM = NetM(NetSP_path, sample_rate, img_height, img_width, down_size, batch_size, temperature_init)
        self.netE = ResNet(layers=18, decoder='deconv2', output_size=(img_height, img_width), in_channels=4, pretrained=True)
            
        self.netE.load_state_dict(torch.load(NetE_path))

    def forward(self, img_rgb, img_depth, is_soft):

        sparse_depth, sampling_mask, pooled_xy_tensor, curr_spixl_map, prob = self.netM(img_rgb, img_depth)
        
        rgb_sparse_d_input = torch.cat((img_rgb, sparse_depth), 1) # white input
        
        x_recon = self.netE(rgb_sparse_d_input)
        
        return x_recon, sparse_depth, sampling_mask, pooled_xy_tensor, curr_spixl_map, prob
    
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=10):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep
        
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch NetRGBM-NetE')
# parser.add_argument('--sample_rate', type=int, default=0.01, help="sample rate for pixel interpolation")
# parser.add_argument('--down_size', type=int, default=10, help='grid cell size for superpixel training')
# parser.add_argument('--temperature', type=float, default=1.0, help='grid cell size for superpixel training')
# parser.add_argument('--img_width', type=int, default=960, help="default train image width")
# parser.add_argument('--img_height', type=int, default=240, help="default train image height")
# parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
# parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
# parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0002')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
# parser.add_argument('--beta', type=float, default=0.999, help='Adam momentum term. Default=0.999')
# parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight_decay for SGD/ADAM')
# # parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum term. Default=0.5')
# # parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
# parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
# parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--train_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/train.csv", help='path to train_csv')
# parser.add_argument('--val_data_csv_path', type=str, default="/home/dqq/Data/KITTI/inpainted/val.csv", help='path to val_csv')
# parser.add_argument('--path_to_save', type=str, default="epochs_NetM_SparseD_wd", help='path to save trained models')
# parser.add_argument('--path_to_tensorboard_log', type=str, default="tensorBoardRuns/NetM-SP-SparseD-linear-bilinear-clip-batch-8-240x960-crop-default-epoch-100-lr-00001-decay-ADAM-c-001-L1-loss-07-26-2020", help='path to tensorboard logging')
# # parser.add_argument('--path_to_NetE_pre', type=str, default="epochs_S2D_SparseD_wd_2020-07-24 22:21:05.290791/model_epoch_100.pth", help='path to tensorboard logging')
# parser.add_argument('--path_to_NetE_pre', type=str, default="epochs_S2D_RGBSparseD_wd_2020-10-17 23:10:45.482731/model_epoch_100.pth", help='path to tensorboard logging')
# parser.add_argument('--path_to_NetSP_pre', type=str, default="epochs_SuperPixeFCN_color_2020-10-17 23:12:09.821133/model_epoch_100.pth", help='path to tensorboard logging')
# # parser.add_argument('--path_to_NetME', type=str, default="epochs_NetM_SparseD_wd_2020-09-29 00:44:59.001504/model_tmp.pth", help='path to tensorboard logging')
# parser.add_argument('--device_ids', type=list, default=[0, 1], help='path to tensorboard logging')
# parser.add_argument('--nef', type=int, default=64, help='number of encoder filters in first conv layer')



# opt = parser.parse_args()

# print(opt)


# print('===> Building model...')
# modelME = NetME_RGBSparseD2Dense(opt.path_to_NetE_pre, opt.path_to_NetSP_pre, opt.sample_rate, opt.img_height, opt.img_width, opt.down_size, opt.batch_size, opt.temperature)
# # modelME.load_state_dict(torch.load(opt.path_to_NetME))
# print(modelME)

# # input_rgb = cv2.imread("/home/dqq/Data/KITTI/inpainted/val/2011_09_26_drive_0002_sync/rgb_image03_0000000005.png")
# # input_depth = cv2.imread("/home/dqq/Data/KITTI/inpainted/val/2011_09_26_drive_0002_sync/d_image03_0000000005.png", cv2.IMREAD_UNCHANGED)

# input_rgb = cv2.imread("/home/dqq/Data/KITTI/inpainted/train/2011_09_28_drive_0057_sync/rgb_image03_0000000045.png")
# input_depth = cv2.imread("/home/dqq/Data/KITTI/inpainted/train/2011_09_28_drive_0057_sync/d_image03_0000000045.png", cv2.IMREAD_UNCHANGED)


# input_rgb = input_rgb.astype(np.float32)/255.0
# input_depth = input_depth.astype(np.float32)/256.0

# def bottom_crop_np(img_rgb, img_ss, crop_height, crop_width):
#     img_height, img_width, _ = img_rgb.shape
    
#     i = img_height - crop_height
#     j = int(round((img_width - crop_width) / 2.))
    
#     img_rgb = img_rgb[i:i+crop_height, j:j+crop_width,:]
#     img_ss = img_ss[i:i+crop_height, j:j+crop_width]
    
#     return img_rgb, img_ss

# input_rgb, input_depth = bottom_crop_np(input_rgb, input_depth, opt.img_height, opt.img_width)

# input_rgb = torch.from_numpy(input_rgb.transpose((2, 0, 1)))
# input_depth = torch.from_numpy(input_depth)

# input_rgb = input_rgb.unsqueeze(0)

# input_depth = input_depth.unsqueeze(0)
# input_depth = input_depth.unsqueeze(0)

# # input_rgb = torch.rand(4, 3, 240, 960)
# # input_depth = torch.ones(4, 1, 240, 960)


# modelME = modelME.cuda()    
# input_rgb = input_rgb.cuda()
# input_depth = input_depth.cuda()

# input_rgb = torch.cat((input_rgb, input_rgb), dim=0)
# input_depth = torch.cat((input_depth, input_depth), dim=0)

# # corrupt_mask, image_recon = modelME(input_rgb, input_depth)
# image_recon, sparse_depth, corrupt_mask, pooled_xy_tensor, curr_spixl_map, prob = modelME(input_rgb, input_depth, True)

# image_recon = image_recon.data.cpu().numpy()
# corrupt_mask = corrupt_mask.data.cpu().numpy()
# corrupt_mask_soft = corrupt_mask

# image_recon = np.squeeze(image_recon)
# corrupt_mask = np.squeeze(corrupt_mask)
# corrupt_mask_soft = np.squeeze(corrupt_mask_soft)

# corrupt_mask_sparsity = np.sum(corrupt_mask)/(corrupt_mask.shape[0]*corrupt_mask.shape[1]*corrupt_mask.shape[2])

# print(f"corrupt_mask_sparsity: {corrupt_mask_sparsity}")

# corrupt_mask = corrupt_mask[0,:,:]
# corrupt_mask = np.repeat(corrupt_mask[:, :, np.newaxis], 3, axis=2)
# corrupt_mask = corrupt_mask.astype(np.float32)

# draw_grid(corrupt_mask, line_color=(0, 1, 0), thickness=1, type_=cv2.LINE_AA, pxstep=10)

# cv2.imwrite("/tmp/corrupt_mask.png", corrupt_mask.astype(np.uint8)*255)


# corrupt_mask_soft = corrupt_mask_soft[0,:,:]
# corrupt_mask_soft = np.repeat(corrupt_mask_soft[:, :, np.newaxis], 3, axis=2)
# corrupt_mask_soft = corrupt_mask_soft.astype(np.float32)

# # draw_grid(corrupt_mask_soft, line_color=(0, 1, 0), thickness=1, type_=cv2.LINE_AA, pxstep=10)

# cv2.imwrite("/tmp/corrupt_mask_soft.png", corrupt_mask_soft.astype(np.uint8)*255)
