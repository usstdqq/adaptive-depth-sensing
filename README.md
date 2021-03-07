# Adaptive Depth Sampling using Deep Learning
A PyTorch implementation of the paper:
[Adaptive Image Sampling using Deep Learning and its Application on X-Ray Fluorescence Image Reconstruction] [[Arxiv Preprint]](https://arxiv.org/abs/1812.10836) [[IEEE Transactions on Multimedia]](https://ieeexplore.ieee.org/document/8930037) 

## Introduction

<img src='figs/pipeline.png' width=800>

The proposed pipeline contains two submodules, adaptive depth sampling (N etM ) and depth reconstruction (N etE). The binary adaptive sampling mask is generated based on the RGB image. Then, the LiDAR samples the scene based on the binary sampling mask and generate the sampled sparse depth map. Finally, both RGB image and sampled sparse depth map are applied to estimate the dense depth map.

<img src='figs/intro_example.png' width=400>

LiDAR systems is able to capture accurate sparse depth map (bottom). Reducing the number of samples is able to increase the capture framerate. RGB image (top) can be applied to fuse with the captured sparse depth data and estimate a dense depth map. We demonstrate that choosing the sampling location is im-
portant to the accuracy of the estimated depth map. Under 0.25% sampling rate (with respect to the RGB image), using the same depth estimation method, the depth map estimated from the adaptively sampled sparse depth (third row) is more accurate than the depth map estimated from random samples (second row).

<img src='figs/expo_results.png' width=800>


## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org/)
- tqdm
```
pip install tqdm
```
- opencv
```
conda install -c conda-forge opencv
```
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
```
pip install tensorboard_logger
```
- h5py
```
conda install h5py
```

## Datasets

### Train and Val Dataset
The train and val datasets are sampled from [ImageNet](http://www.image-net.org/).
Train dataset has 100000 images. Val dataset has 1000 images.
Download the datasets from [here](https://drive.google.com/file/d/1RNfvuZKdf8MZAb1zzVgsFSlX36oc1uPA/view?usp=sharing), 
and then extract it into `$data` directory. Modify the path of `$data` directory in line#48 of file train_NetE.py and line#48 of file train_NetM.py.

### Test Image Dataset
The test image dataset are sampled from [ImageNet](http://www.image-net.org/). It contains 100 images. It is stored in file data_val_100.h5 .

## Usage

### Train NetE

Run
```
python train_NetE.py
```
to train the image inpainting network NetE.

### Train NetM

After NetE is trained, modify the file name of trained NetE in line#29 of file train_NetM.py and run
```
python train_NetM.py
```
to train the adaptive image sampling network NetM.

To visualize the training process, run
```
tensorboard --logdir tensorBoardRuns
```

### Test NetE
```
python test_NetE_h5.py
```
The output reconstructed images are in `results/netE_results` directory.

### Test NetM
```
python test_NetM_h5.py
```
The output reconstructed images are in `results/netM_results` directory.

