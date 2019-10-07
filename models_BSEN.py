"""
BSEN
2019
Author:
        Wan-Ting Hsieh       cclee@ee.nthu.edu.tw

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from math import pi
from copy import deepcopy
import torch.autograd as autograd



#%%
class AutoEncoder(nn.Module):
    def __init__(self,ks1, ks2, ks3):
        super(AutoEncoder, self).__init__()


        self.encoder = nn.Sequential(
            ## conv formula: (N+2*P-K)/S
            nn.Conv3d(1, ks1, kernel_size=3, stride=1,padding=1), #4,64,80,64
            nn.BatchNorm3d(ks1),
#            nn.SELU(),
            nn.MaxPool3d(2),

            nn.Conv3d(ks1, ks2, kernel_size=3, stride=1,padding=1), # 8,32,40,32
            nn.BatchNorm3d(ks2),
#            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(ks2, ks3, kernel_size=3, stride=1,padding=1), # 16,16,20,16
            nn.BatchNorm3d(ks3),
            nn.SELU(),
            nn.MaxPool3d(2),)

#
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(ks3, ks2, kernel_size=3, stride=1,padding=1), # 8,64,465
            nn.BatchNorm3d(ks2),
#            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(ks2, ks1, kernel_size=3, stride=1,padding=1), #8,64,4745
            nn.BatchNorm3d(ks1),
#            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(ks1, 1, kernel_size=3, stride=1,padding=1),
#            nn.BatchNorm3d(1),
            nn.ReLU())

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x ,x1



class ChannelModel(nn.Module):
    def __init__(self,hidden_n,dec_len,model,channel_to_decode):
        super(ChannelModel, self).__init__()
        self.dec_upsample_part = nn.Upsample(scale_factor=2, mode='trilinear')

        self.dec_channel_part_dec1 = nn.ConvTranspose3d(1, hidden_n, kernel_size=3, stride=1,padding=1)        #hidden n = k-1
        ori_channel_weight = (list(model.dec2)[1].weight.data)[channel_to_decode,:,:]
        ori_channel_bias = (list(model.dec2)[1].bias.data)
        self.dec_channel_part_dec1.state_dict()['weight'].data.copy_(ori_channel_weight.reshape(1,ori_channel_weight.shape[0],ori_channel_weight.shape[1],ori_channel_weight.shape[2],ori_channel_weight.shape[3]))
        self.dec_channel_part_dec1.state_dict()['bias'].data.copy_(ori_channel_bias)
        self.dec1_part = nn.Sequential(*list(model.dec2)[2:]).cpu()

        self.dec2_part = nn.Sequential(*list(model.dec3)).cpu()
        self.dec3_part = nn.Sequential(*list(model.dec4)).cpu()
#        self.dec4_part = nn.Sequential(*list(model.dec4)).cpu()


    def forward(self, x):
        x1 = self.dec_upsample_part(x)
        x1 = self.dec_channel_part_dec1(x1)
        x1 = self.dec1_part(x1)

        x1 = self.dec2_part(x1)
        x1 = self.dec3_part(x1)

        return x1



#%% mmd loss
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def loss_mmd(z):
    true_samples = torch.randn_like(z,requires_grad=False)
    mmd = compute_mmd(true_samples, z)
    return mmd
