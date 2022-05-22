"""
	Very Deep Convolutional Networks for Large-Scale Image Recognition[University of Oxford]
"""
import torch
import torch.nn as nn
from torchsummary import summary
from CVBenchmark.torch.register import modelRegister

VGGConfiguration = {
	'VGG13':[3, 3, 'M', 3, 3, 'M', 3, 3, 'M', 3, 3, 'M', 3, 3, 'M'],
	'VGG16_Conv1':[3, 3, 'M', 3, 3, 'M', 3, 3, 1, 'M', 3, 3, 1, 'M', 3, 3, 1, 'M'],
	'VGG16':[3, 3, 'M', 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M'],
	'VGG19':[3, 3, 'M', 3, 3, 'M', 3, 3, 3, 3, 'M', 3, 3, 3, 3, 'M', 3, 3, 3, 3, 'M']
}

@modelRegister.register
class VGG(nn.Module):
	def __init__(self, VGGName='VGG19', num_classes=1000):
		super(VGG, self).__init__()
		self.feature = self._makeLayer(VGGName)
		self.classifier = nn.Linear(in_features=4096, out_features=num_classes)

	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _makeLayer(self, VGGName):
		layers = []
		inChannel = 3
		outChannel = 64
		for l in VGGConfiguration[VGGName]:
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
			layers.append(nn.Dropout(0.5))
			inFeatures = 4096
		return nn.Sequential(*layers)

if __name__=="__main__":
    model = VGG('VGG19')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    summary(model, (3, 224, 224))
