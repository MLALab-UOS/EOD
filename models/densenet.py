# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .route import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, normalizer = None,
                 out_classes = 100, k=None, info=None):
        super(DenseNet3, self).__init__()

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = int(n/2)
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)


        if k is None:
            self.fc = nn.Linear(in_planes, num_classes)
        else:
            self.fc = RouteFcUCPruned(in_planes, num_classes, topk=k, info=info)
        #     # self.fc = RouteDropout(in_planes, num_classes, p=k)

        self.in_planes = in_planes
        self.normalizer = normalizer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        return out

    def forward_virtual(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, 8)
        # breakpoint()
        out = out.view(-1, self.in_planes)
        return self.fc(out), out

    def features(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:, 0, :, :] = (x[:, 0, :, :] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:, 1, :, :] = (x[:, 1, :, :] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:, 2, :, :] = (x[:, 2, :, :] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    # function to extact the multiple features
    def feature_list(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = self.trans1(self.block1(out))
        out_list.append(out)
        out = self.trans2(self.block2(out))
        out_list.append(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)

        return self.fc(out), out_list

    def intermediate_forward(self, x, layer_index):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        if layer_index == 1:
            out = self.trans1(self.block1(out))
        elif layer_index == 2:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
        elif layer_index == 3:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        penultimate = self.relu(self.bn1(out))
        out = F.avg_pool2d(penultimate, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out), penultimate