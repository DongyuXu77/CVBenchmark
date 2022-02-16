"""
	MobileNetV2: Inverted Residuals and Linear Bottlenecks[Google Inc.]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
MobileNetV2Configuration = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1), 1280]	#(expansion_factor, channel, repeated_time, stride)

class bottleneck(nn.Module):
	def __init__(self, inChannels, outChannels, stride, expansion_factor):
		super(bottleneck, self).__init__()
		self.stride = stride
		self.plain = self._makeLayer(inChannels, outChannels, stride, expansion_factor)
		self.identity = self._shortcut(inChannels, outChannels)

	def forward(self, x):
		out = self.plain(x)
		if self.stride==1:
			out = out+self.identity(x)
		out = F.relu(out)
		return out

	def _makeLayer(self, inChannels, outChannels, stride, expansion_factor):
		layers = []
		midChannels = expansion_factor*inChannels
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=(1, 1)))
		layers.append(nn.BatchNorm2d(midChannels))
		layers.append(nn.ReLU6())
		layers.append(nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=(3, 3), stride=stride, padding=1))
		layers.append(nn.BatchNorm2d(midChannels))
		layers.append(nn.ReLU6())
		layers.append(nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=(1,1)))
		layers.append(nn.BatchNorm2d(outChannels))
		return nn.Sequential(*layers)

	def _shortcut(self, inChannels, outChannels):
		layers = []
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1)))
		layers.append(nn.BatchNorm2d(outChannels))
		return nn.Sequential(*layers)

class MobileNetV2(nn.Module):
	def __init__(self, num_classes=1000):
		super(MobileNetV2, self).__init__()
		self.feature = self._makeLayer()
		self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

	def forward(self, x):	
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self):
		layers = []
		layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1))
		layers.append(nn.BatchNorm2d(32))
		inChannels = 32
		for config in MobileNetV2Configuration:
			if not isinstance(config, int):
				expansion_factor, outChannels, repeat_time, stride = config
				for l in range(repeat_time):
					if l == 0:
						layers.append(bottleneck(inChannels, outChannels, stride, expansion_factor))
					else:
						layers.append(bottleneck(inChannels, outChannels, 1, expansion_factor))
					inChannels = outChannels
			else:
				layers.append(nn.Conv2d(in_channels=inChannels, out_channels=config, kernel_size=(1, 1)))
				layers.append(nn.BatchNorm2d(config))
				layers.append(nn.ReLU())
				layers.append(nn.AvgPool2d(kernel_size=(7, 7)))
		return nn.Sequential(*layers)
	
		
if __name__=="__main__":
	model = MobileNetV2()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	summary(model, (3, 224, 224))
