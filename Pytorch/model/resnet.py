"""
	Deep Residual Learning for Image Recognition[Microsoft Research]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

resnetConfiguration = {
    'resnet18':{
        'block': 'buildingblock',
        'num_blocks': [2, 2, 2, 2]
    },
    'resnet34':{
        'block': 'buildingblock',
        'num_blocks': [3, 4, 6, 3]
    },
    'resnet50':{
        'block': 'bottleneck',
        'num_blocks': [3, 4, 6, 3]
    },
    'resnet101':{
        'block': 'bottleneck',
        'num_blocks': [3, 4, 23, 3]
    },
    'resnet152':{
        'block': 'bottleneck',
        'num_blocks': [3, 8, 36, 3]
    }
}

class buildingblock(nn.Module):
    expansion = 1

    def __init__(self, inPlanes, planes, stride=1):
        super(buildingblock, self).__init__()
        self.plain = self._makeLayers(inPlanes, planes, stride)
        self.identity = self._shortcut(inPlanes, planes, stride)

    def forward(self, x):
        out = self.plain(x)
        out = out+self.identity(x)
        out = F.relu(out)
        return out

    def _makeLayers(self, inPlanes, planes, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels=inPlanes, out_channels=planes, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(self.expansion*planes))
        return nn.Sequential(*layers)

    def _shortcut(self, inPlanes, planes, stride):
        layers = []
        if inPlanes!=self.expansion*planes or stride!=1:
            layers.append(nn.Conv2d(in_channels=inPlanes, out_channels=self.expansion*planes, kernel_size=1, stride=stride))
            layers.append(nn.BatchNorm2d(self.expansion*planes))
        return nn.Sequential(*layers)

class bottlenect(nn.Module):
    expansion = 4

    def __init__(self, inPlanes, planes, stride=1):
        super(bottlenect, self).__init__()
        self.plain = self._makeLayers(inPlanes, planes, stride)
        self.identity = self._shortcut(inPlanes, planes, stride)

    def forward(self, x):
        out = self.plain(x)
        out = out+self.identity(x)
        out = F.relu(out)
        return out

    def _makeLayers(self, inPlanes, planes, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels=inPlanes, out_channels=planes, kernel_size=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=1))
        layers.append(nn.BatchNorm2d(self.expansion*planes))
        return nn.Sequential(*layers)

    def _shortcut(self, inPlanes, planes, stride):
        layers = []
        if inPlanes!=self.expansion*planes or stride!=1:
            layers.append(nn.Conv2d(in_channels=inPlanes, out_channels=self.expansion*planes, kernel_size=1 ,stride=stride))
            layers.append(nn.BatchNorm2d(self.expansion*planes))
        return nn.Sequential(*layers)


class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(resnet, self).__init__()
        self.inPlanes = 64

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature1 = self._makeLayers(block, 64, num_blocks[0], stride=1)
        self.feature2 = self._makeLayers(block, 128, num_blocks[1], stride=2)
        self.feature3 = self._makeLayers(block, 256, num_blocks[2], stride=2)
        self.feature4 = self._makeLayers(block, 512, num_blocks[3], stride=2)
        self.avgPool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

    def forward(self, x):
        x = self.maxPool(F.relu(self.bn(self.conv(x))))
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _makeLayers(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.inPlanes, planes, s))
            self.inPlanes = planes*block.expansion
        return nn.Sequential(*layers)

if __name__=="__main__":
    model = resnet(buildingblock, [3, 4, 6, 3], 1000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    summary(model, (3, 224, 224))
