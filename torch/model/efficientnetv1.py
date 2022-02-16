"""
	EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks[Google Inc.]
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
EfficientNetV1Configuration = [(3, 1, 16, 1), (3, 2, 24, 2), (5, 2, 40, 2), (3, 2, 80, 3), (5, 2, 112, 3), (5, 2, 192, 4), (3, 1, 320, 1)]	#(kernel_size, stride, out_Channels, repeat_time(stage))

class bottleneck(nn.Module):
	def __init__(self, inChannels, outChannels, kernel_size, stride):
		super(bottleneck, self).__init__()
		self.stride = stride
		self.plain = self._makeLayer(inChannels, outChannels, kernel_size, stride)
		self.identity = self._shortcut(inChannels, outChannels, stride)
	
	def forward(self, x):
		out = self.plain(x)
		if self.stride==1:
			out = out+self.identity(x)
		out = F.relu(out)
		return out
	
	def _makeLayer(self, inChannels, outChannels, kernel_size, stride, expansion_factor=6):
		layers = []
		midChannels = inChannels*expansion_factor
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=1))
		layers.append(nn.BatchNorm2d(midChannels))
		layers.append(nn.ReLU6())
		layers.append(nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
		layers.append(nn.BatchNorm2d(midChannels))
		layers.append(nn.ReLU6())
		layers.append(nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=1))
		layers.append(nn.BatchNorm2d(outChannels))
		return nn.Sequential(*layers)

	def _shortcut(self, inChannels, outChannels, stride):
		layers = []
		if inChannels!=outChannels:
			layers.append(nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1, stride=stride))
			layers.append(nn.BatchNorm2d(outChannels))
		return nn.Sequential(*layers)

class EfficientNetV1(nn.Module):
	def __init__(self, num_classes=1000):
		super(EfficientNetV1, self).__init__()
		self.feature = self._makeLayer()
		self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self):
		layers = []
		layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
		layers.append(nn.BatchNorm2d(32))
		inChannels = 32
		for config in EfficientNetV1Configuration:
			kernel_size, stride, outChannels, repeat_time = config
			for l in range(repeat_time):
				if l==0:
					layers.append(bottleneck(inChannels, outChannels, kernel_size, stride))
					inChannels = outChannels
				else:
					layers.append(bottleneck(inChannels, outChannels, kernel_size, 1))
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=1280, kernel_size=1))
		layers.append(nn.AvgPool2d(kernel_size=7))
		return nn.Sequential(*layers)

if __name__=="__main__":
	model = EfficientNetV1()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	summary(model, (3, 224, 224))
