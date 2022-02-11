"""
	Very Deep Convolutional Networks for Large-Scale Image Recognition[University of Oxford]
"""
import torch
import torch.nn as nn
from torchsummary import summary

vggConfiguration = {
	'vgg13':[3, 3, 'M', 3, 3, 'M', 3, 3, 'M', 3, 3, 'M', 3, 3, 'M'],
	'vgg16_Conv1':[3, 3, 'M', 3, 3, 'M', 3, 3, 1, 'M', 3, 3, 1, 'M', 3, 3, 1, 'M'],
	'vgg16':[3, 3, 'M', 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M'],
	'vgg19':[3, 3, 'M', 3, 3, 'M', 3, 3, 3, 3, 'M', 3, 3, 3, 3, 'M', 3, 3, 3, 3, 'M']
}

class vgg(nn.Module):
	def __init__(self, vggName='vgg19'):
		super(vgg, self).__init__()
		self.feature = self._makeLayer(vggName)
		self.classifier = nn.Linear(in_features=4096, out_features=1000)

	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self, vggName):
		layers = []
		inChannel = 3
		outChannel = 64
		for l in vggConfiguration[vggName]:
			if l=='M':
				layers.append(nn.MaxPool2d(kernel_size=2))
				if outChannel<512:
					outChannel= outChannel*2
				else:
					layers.append(nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, padding=1))
					inChannel = outChannel
		layers.append(nn.Flatten())
		inFeatures = 25088 # 7*7*512
		for l in range(2):
			layers.append(nn.Linear(in_features=inFeatures, out_features=4096))
			inFeatures = 4096
		return nn.Sequential(*layers)

if __name__=="__main__":
	model = vgg('vgg19')
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	summary(model, (3, 224, 224))
