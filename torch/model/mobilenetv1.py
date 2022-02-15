"""
	MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[Google Inc.]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
MobileNetV1Configuration = {
	'layers' : [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024],
	'width_multipler' : 0.75, #one of 0.25/0.75/1
	'resolution_multiplier' : 1#one of 0.7414/1
}


class MobileNetV1(nn.Module):
	def __init__(self):
		super(MobileNetV1, self).__init__()
		self.feature = self._makeLayer()
		self.classifier = nn.Linear(in_features=int(1024*MobileNetV1Configuration['width_multipler']), out_features=1000)

	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self):
		layers = []
		width = MobileNetV1Configuration['width_multipler']
		layers.append(nn.Conv2d(in_channels=3, out_channels=int(width*32), kernel_size=(3, 3), stride=2))
		layers.append(nn.BatchNorm2d(int(width*32)))
		inChannels = int(32*MobileNetV1Configuration['width_multipler'])
		for config in MobileNetV1Configuration['layers']:
			outChannels = config if isinstance(config, int) else config[0]
			stride = 1 if isinstance(config, int) else config[1]
			layers.append(self.depthwiseSeparableConvolution(inChannels, int(outChannels*width), stride))
			inChannels = int(outChannels*width)
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
	model = MobileNetV1()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	summary(model, (3, 224, 224))
