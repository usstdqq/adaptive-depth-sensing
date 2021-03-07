import os
from PIL import Image
import numpy as np
import torch
import random
import pandas as pd
import cv2

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.tif'])

def random_horizontal_flip(img_rgb, img_ss):
    if np.random.uniform(0,1) > 0.5:
        img_rgb = img_rgb[:,::-1,:]
        img_ss = img_ss[:,::-1]
    return img_rgb, img_ss

def random_vertical_flip(img_rgb, img_ss):
    if np.random.uniform(0,1) > 0.5:
        img_rgb = img_rgb[::-1,:,:]
        img_ss = img_ss[::-1,:]
    return img_rgb, img_ss

def random_crop_np(img_rgb, img_ss, crop_height, crop_width):
    img_height, img_width, _ = img_rgb.shape
    
    start_height = np.random.randint(0, img_height-crop_height)
    start_width = np.random.randint(0, img_width-crop_width)
    
    img_rgb = img_rgb[start_height:start_height+crop_height, start_width:start_width+crop_width,:]
    img_ss = img_ss[start_height:start_height+crop_height, start_width:start_width+crop_width]
    
    return img_rgb, img_ss


def bottom_crop_np(img_rgb, img_ss, crop_height, crop_width):
    img_height, img_width, _ = img_rgb.shape
    
    i = img_height - crop_height
    j = int(round((img_width - crop_width) / 2.))
    
    img_rgb = img_rgb[i:i+crop_height, j:j+crop_width,:]
    img_ss = img_ss[i:i+crop_height, j:j+crop_width]
    
    return img_rgb, img_ss


class KITTI_SS_Dataset(Dataset):
    def __init__(self, path_to_csv: str, is_train: bool, img_height: int, img_width: int):
        self.path_to_csv = path_to_csv
        
        self.df = pd.read_csv(self.path_to_csv, header=None)
        
        self.is_train = is_train
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.transform = get_transform(is_train)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        
        image_rgb = cv2.imread(self.df.iloc[index][0])[:, :, ::-1].astype(np.float32)
        image_ss = cv2.imread(self.df.iloc[index][1])[:,:,:1].astype(np.int)
        
        image_rgb = image_rgb/255.0
        
        if self.is_train:
            image_rgb, image_ss = random_crop_np(image_rgb, image_ss, self.img_height, self.img_width)
            image_rgb, image_ss = random_horizontal_flip(image_rgb, image_ss)
            image_rgb, image_ss = random_vertical_flip(image_rgb, image_ss)
        else:
            image_rgb, image_ss = bottom_crop_np(image_rgb, image_ss, self.img_height, self.img_width)
            
        image_rgb = torch.from_numpy(image_rgb.copy()).float()
        image_rgb = image_rgb.permute(2, 0, 1)
        
        image_ss = torch.from_numpy(image_ss.copy()).int()
        image_ss = image_ss.permute(2, 0, 1)
        # image_ss.unsqueeze()
        
        return image_rgb, image_ss


class KITTI_Dataset(Dataset):
    def __init__(self, path_to_csv: str, is_train: bool):
        self.path_to_csv = path_to_csv
        
        self.df = pd.read_csv(self.path_to_csv)
        
        self.is_train = is_train
        
        self.transform = get_transform(is_train)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # print(self.df.iloc[index][0])
        # print(self.df.iloc[index][1])
        image_rgb = Image.open(self.df.iloc[index][0])
        image_depth = Image.open(self.df.iloc[index][1])
        
        # print(np.asarray(image_depth))
        
        sample = {'rgb': image_rgb, 'gt': image_depth}
        
        sample = self.transform(sample)
        
        image_rgb = sample['rgb']
        depth_map = sample['gt']
        
        # print(np.asarray(depth_map))
        
        depth_mask = depth_map > 0
        
        return image_rgb, depth_map, depth_mask

class ToTensorKITTI(object):
    def __call__(self, sample):
        image, depth = sample['rgb'], sample['gt']

        # image = image.resize((1216, 352))
        # image = bottom_crop(image, (1216, 352))
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float()

        # image = bottom_crop(image, (228, 912))
        # depth = bottom_crop(depth, (228, 912))
        
        image = bottom_crop(image, (240, 960))
        depth = bottom_crop(depth, (240, 960))

        # put in expected range

        return {'rgb': image, 'gt': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False).copy())
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False).copy())
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img.float().div(256)


def get_transform(is_train):
    if is_train:
        return transforms.Compose([
            RandomHorizontalFlip(),
            RandomChannelSwap(0.5),
            ToTensorKITTI()
        ])
    else:
        return transforms.Compose([
            ToTensorKITTI()
        ])


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['rgb'], sample['gt']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'rgb': image, 'gt': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['rgb'], sample['gt']
        if not _is_pil_image(image): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError(
            'img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(
                self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'rgb': image, 'gt': depth}


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def bottom_crop(img, output_size):
    h = img.shape[1]
    w = img.shape[2]
    th, tw = output_size
    i = h - th
    j = int(round((w - tw) / 2.))

    if img.dim() == 3:
        return img[:, i:i + th, j:j + tw]
    elif img.ndim == 2:
        return img[i:i + th, j:j + tw]
    

# import argparse
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch SuperPixel')
# parser.add_argument('--sample_rate', type=int, default=0.01, help="sample rate for pixel interpolation")
# parser.add_argument('--train_img_width', type=int, default=912, help="default train image width")
# parser.add_argument('--train_img_height', type=int, default=228, help="default train image height")
# parser.add_argument('--input_img_width', type=int, default=912, help="default train image width")
# parser.add_argument('--input_img_height', type=int, default=228, help="default train image height")
# parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
# parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
# parser.add_argument('--lr', type=float, default=0.00005, help='Learning Rate. Default=0.0002')
# parser.add_argument('--downsize', type=float, default=10, help='grid cell size for superpixel training ')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
# parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight_decay')
# parser.add_argument('--bias_decay', type=float, default=0.0, help='weight_decay')
# parser.add_argument('--beta', type=float, default=0.999, help='Adam momentum term. Default=0.999')
# # parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
# parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
# parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--train_data_path', type=str, default="/home/dqq/Data/KITTI/data_semantics/train.csv", help='path to train data')
# parser.add_argument('--val_data_path', type=str, default="/home/dqq/Data/KITTI/data_semantics/train.csv", help='path to val data')
# parser.add_argument('--path_to_save', type=str, default="epochs_SuperPixeFCN", help='path to save trained models')
# parser.add_argument('--path_to_tensorboard_log', type=str, default="tensorBoardRuns/SuperPixelFCN-batch-4-bottom-crop-default-epoch-100-lr-000005-ADAN-c-001-seg-loss-07-22-2020", help='path to tensorboard logging')
# parser.add_argument('--device_ids', type=list, default=[0, 1], help='path to tensorboard logging')

# opt = parser.parse_args()
# train_set = KITTI_SS_Dataset(path_to_csv = opt.train_data_path, is_train=True,  img_height=opt.train_img_height, img_width=opt.train_img_width)
# print(train_set.__len__())
# aa, bb= train_set.__getitem__(0)