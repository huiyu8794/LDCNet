import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import torchvision.models as models
from models.Learnable_Res18 import resnet18


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class LDCNet(nn.Module):
    def __init__(self):
        super(LDCNet, self).__init__()
        self.backbone = FE_Res18_learnable()
        self.LiveEstor = Estmator_learnable()
        self.SpoofEstor = Estmator_learnable()
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature, catfeat = self.backbone(input) 
        live_Pred = self.LiveEstor(catfeat)
        spoof_Pred = self.SpoofEstor(catfeat)
        classifier_out = self.classifier(feature, norm_flag)
        
        return classifier_out, live_Pred, spoof_Pred, feature 


class FE_Res18_learnable(nn.Module):
    def __init__(self):
        super(FE_Res18_learnable, self).__init__()
        model_resnet = resnet18()
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        model_resnet.load_state_dict(state_dict, strict=False)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        feature = self.conv1(input) 
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        re_feature1 = F.adaptive_avg_pool2d(feature1, 32)
        re_feature2 = F.adaptive_avg_pool2d(feature2, 32)
        re_feature3 = F.adaptive_avg_pool2d(feature3, 32)
        catfeat = torch.cat([re_feature1, re_feature2, re_feature3], 1)

        feature = self.layer4(feature3)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(feature, feature_norm)

        return feature, catfeat



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out



class conv3x3_learn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3_learn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff


class Estmator_learnable(nn.Module):
    def __init__(self, in_channels=448, out_channels=1, conv3x3=conv3x3_learn):
        super(Estmator_learnable, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class ResNet_Amodel(nn.Module):
    def __init__(self, ):
        super(ResNet_Amodel, self).__init__()
        self.FeatExtor_LS = models.resnet18(pretrained=True)
        self.FC_LS = torch.nn.Linear(1000, 2)

    def forward(self, x): 
        x = self.FeatExtor_LS(x)
        x = self.FC_LS(x)
        return x