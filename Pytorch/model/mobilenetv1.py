import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
mobilenetv1Configuration = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

class mobilenetv1(nn.Module):
	def __init__(self):
		super(mobilenetv1, self).__init__()
		self.feature = self._makeLayer()
		self.classifier = nn.Linear(in_features=1024, out_features=1000)

	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self):
		layers = []
		layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2))
		layers.append(nn.BatchNorm2d(32))
		inChannels = 32
		for config in mobilenetv1Configuration:
			outChannels = config if isinstance(config, int) else config[0]
			stride = 1 if isinstance(config, int) else config[1]
			layers.append(self.depthwiseSeparableConvolution(inChannels, outChannels, stride))
			inChannels = outChannels
		layers.append(nn.AvgPool2d(kernel_size=(7, 7)))
		return nn.Sequential(*layers)

	def depthwiseSeparableConvolution(self, inChannels, outChannels, stride):
		layers = []
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=inChannels, kernel_size=(3, 3), stride=stride, padding=1))
		layers.append(nn.BatchNorm2d(inChannels))
		layers.append(nn.ReLU())
		layers.append(nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), stride=1))
		layers.append(nn.BatchNorm2d(outChannels))
		layers.append(nn.ReLU())
		return nn.Sequential(*layers)

if __name__=="__main__":
	model = mobilenetv1()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	summary(model, (3, 224, 224))
