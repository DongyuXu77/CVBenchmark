"""
	Deep Residual Learning for Image Recognition[Microsoft Research]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

ResNetConfiguration = {
	'ResNet18':{
		'block': 'buildingblock',
		'num_blocks': [2, 2, 2, 2]
	},
	'ResNet34':{
		'block': 'buildingblock',
		'num_blocks': [3, 4, 6, 3]
	},
	'ResNet50':{
		'block': 'bottleneck',
		'num_blocks': [3, 4, 6, 3]
	},
	'ResNet101':{
		'block': 'bottleneck',
		'num_blocks': [3, 4, 23, 3]
	},
	'ResNet152':{
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

	@classmethod
	def _get_class(cls):
		return cls

class bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inPlanes, planes, stride=1):
		super(bottleneck, self).__init__()
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
	
	@classmethod
	def _get_class(cls):
		return cls

classDict = {'buildingblock':buildingblock._get_class(), 'bottleneck':bottleneck._get_class()}

class ResNet(nn.Module):
	def __init__(self, configuration=ResNetConfiguration['ResNet152'], num_classes=1000):
		super(ResNet, self).__init__()
		self.block = classDict[configuration['block']]
		self.num_blocks = configuration['num_blocks']
		self.inPlanes = 64

		self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
		self.bn = nn.BatchNorm2d(num_features=64)
		self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.feature1 = self._makeLayers(self.block, 64, self.num_blocks[0], stride=1)
		self.feature2 = self._makeLayers(self.block, 128, self.num_blocks[1], stride=2)
		self.feature3 = self._makeLayers(self.block, 256, self.num_blocks[2], stride=2)
		self.feature4 = self._makeLayers(self.block, 512, self.num_blocks[3], stride=2)
		self.avgPool = nn.AvgPool2d(kernel_size=7)
		self.classifier = nn.Linear(in_features=512*self.block.expansion, out_features=num_classes)

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
	model = ResNet(ResNetConfiguration['ResNet152'], 1000)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	summary(model, (3, 224, 224))
