import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock

'''
Reference: `Regularized Fine-Grained Meta Face Anti-Spoofing` (AAAI'20)
- https://github.com/rshaojimmy/RFMetaFAS/blob/master/models/DGFANet.py
'''

class LDCNet_visualize(nn.Module):
    def __init__(self, ):
        super(LDCNet_visualize, self).__init__()

        self.FeatExtor = FeatExtractor()
        self.FeatEmbder = FeatEmbedder()
        self.LiveEstor = Estmator()
        self.SpoofEstor = Estmator()

        self.flatten = nn.Flatten()

    def forward(self, x):
        feat_ext_all, feat, dx1, dx2 = self.FeatExtor(x)
        pred = self.FeatEmbder(feat)
        live_Pred = self.LiveEstor(feat_ext_all)
        spoof_Pred = self.SpoofEstor(feat_ext_all)
        # We only use dx2 feature to visualize
        return pred, live_Pred, spoof_Pred, dx1, dx2



def conv_block(index, in_channels, out_channels, K_SIZE=3, stride=1, padding=1, momentum=0.1, pooling=True):
    # Reference: Regularized fine-grained meta face anti-spoofing (AAAI'20)
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv' + str(index), nn.Conv2d(in_channels, out_channels, \
                                                K_SIZE, stride=stride, padding=padding)),
                ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum, \
                                                   affine=True)),
                ('relu' + str(index), nn.ReLU(inplace=True)),
                ('pool' + str(index), nn.MaxPool2d(2))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv' + str(index), nn.Conv2d(in_channels, out_channels, \
                                                K_SIZE, padding=padding)),
                ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum, \
                                                   affine=True)),
                ('relu' + str(index), nn.ReLU(inplace=True))
            ]))
    return conv


class conv3x3_learn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        super(conv3x3_learn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0),self.conv.weight.size(1)]), requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
    def forward(self, x):
        # Reference: `Searching Central Difference Convolutional Networks for Face Anti-Spoofing` (CVPR'20)
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2) [:, :, None, None]
        
        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff
        

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal


class Estmator(nn.Module):
    def __init__(self, in_channels=384, out_channels=1, conv3x3=conv3x3):
        super(Estmator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128, theta=0.2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64, theta=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, out_channels, theta=0.2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv_learn(nn.Module):
    def __init__(self, in_channels, out_channels, conv3x3=conv3x3_learn):
        super(inconv_learn, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downconv_learn(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, conv3x3=conv3x3_learn):
        super(Downconv_learn, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, conv3x3=conv3x3):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128, theta=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 196, theta=0),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels, theta=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatExtractor, self).__init__()
        self.inc = inconv_learn(in_channels, 64)
        self.down1 = Downconv_learn(64, 128)
        self.down2 = Downconv(128, 128)
        self.down3 = Downconv(128, 128)
        
    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1) 
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)

        return catfeat, dx4, dx1, dx2



class FeatEmbedder(nn.Module):
    def __init__(self, in_channels=128, momentum=0.1):
        super(FeatEmbedder, self).__init__()

        self.momentum = momentum

        self.features = nn.Sequential(
            conv_block(0, in_channels=in_channels, out_channels=128, momentum=self.momentum, pooling=True),
            conv_block(1, in_channels=128, out_channels=256, momentum=self.momentum, pooling=True),
            conv_block(2, in_channels=256, out_channels=512, momentum=self.momentum, pooling=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, 1)

    def forward(self, x, params=None):
        if params == None:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:

            out = F.conv2d(
                x,
                params['features.0.conv0.weight'],
                params['features.0.conv0.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.0.bn0.running_mean'],
                params['features.0.bn0.running_var'],
                params['features.0.bn0.weight'],
                params['features.0.bn0.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)

            out = F.conv2d(
                out,
                params['features.1.conv1.weight'],
                params['features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.1.bn1.running_mean'],
                params['features.1.bn1.running_var'],
                params['features.1.bn1.weight'],
                params['features.1.bn1.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)

            out = F.conv2d(
                out,
                params['features.2.conv2.weight'],
                params['features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.2.bn2.running_mean'],
                params['features.2.bn2.running_var'],
                params['features.2.bn2.weight'],
                params['features.2.bn2.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, 1)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['fc.weight'],
                           params['fc.bias'])
        return out

 
