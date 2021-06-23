import torch
from torch import nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ResNetBody(nn.Module):
    def __init__(self, architecture):
        super(ResNetBody, self).__init__()
        resnet = getattr(models, architecture)(pretrained=True)
        self.in_channels = resnet.layer1[0].conv1.in_channels
        self.out_channels = resnet.fc.in_features
        resnet.conv1 = resnet.bn1 = resnet.relu = resnet.maxpool = Identity()
        resnet.fc = Identity()
        self.resnet = resnet
    
    def forward(self, x):
        return self.resnet(x)

def build_resnetbody(architecture):
    return ResNetBody(architecture)
