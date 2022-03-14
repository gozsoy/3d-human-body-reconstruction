import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from discriminator.util import batch_rodrigues, add_gaussian_noise


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.0):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.relu(x)
        y = self.batch_norm1(y)
        y = self.dropout(y)
        y = self.w1(y)

        y = self.relu(y)
        y = self.batch_norm2(y)
        y = self.dropout(y)
        y = self.w2(y)

        out = x + y

        return out


class ShapeDiscriminator(nn.Module):
    # https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.p = cfg["training"]["disc_start_dropout"]

        self.linear = nn.Sequential(nn.Linear(10, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1),
                                    nn.Sigmoid())
    
    def forward(self, inputs):
        if self.training:
            inputs = add_gaussian_noise(inputs, self.p)

        inputs = self.linear(inputs)
        return inputs


class PoseDiscriminator(nn.Module):
    # https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py
    def __init__(self, cfg):
        super().__init__()
        channels = [9, 32, 32, 32, 1]
        self.cfg = cfg
        self.conv_blocks = nn.Sequential()
        self.p = cfg["training"]["disc_start_dropout"]
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name = 'conv_{}'.format(idx),
                module = nn.Conv2d(in_channels = channels[idx], out_channels = channels[idx + 1], kernel_size = 1, stride = 1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features = channels[l - 2], out_features = 1))

    def forward(self, inputs):
        if self.training:
            inputs = add_gaussian_noise(inputs, self.p)
        inputs = inputs.transpose(1, 2).unsqueeze(2) # 16, 9, 1, 23
        
        internal_outputs = self.conv_blocks(inputs)
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:,:,0,idx]))
        
        return nn.Sigmoid()(torch.cat(o, 1)), internal_outputs


class FullPoseDiscriminator(nn.Module):
    # https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.p = cfg["training"]["disc_start_dropout"]
        
        self.linear = nn.Sequential(nn.Linear(23 * 32, 32),
                                    nn.Linear(32, 1),
                                    nn.Sigmoid())

    def forward(self, inputs):
        if self.training:
            inputs = add_gaussian_noise(inputs, self.p)
        inputs = self.linear(inputs)
        return inputs


class Discriminator(nn.Module):
    # https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py
    def __init__(self, cfg):
        super().__init__()
        self.pose_discriminator = PoseDiscriminator(cfg=cfg)
        self.full_pose_discriminator = FullPoseDiscriminator(cfg=cfg)
        self.shape_discriminator = ShapeDiscriminator(cfg=cfg)

    def forward(self, thetas):
        batch_size = thetas["betas"].shape[0]
        poses = torch.cat((thetas["root_orient"], thetas["pose_body"], thetas["pose_hand"]), 1)
        shapes = thetas["betas"]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return pose_disc_value, full_pose_disc_value, shape_disc_value
