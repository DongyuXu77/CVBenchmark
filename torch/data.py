import torch
import torchvision
from torchvision import datasets, transforms

def dataloader(config):
	if config['dataset'] == 'ImageNet':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])
		transforms_train = transforms.Compose([
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize
		])
		transforms_test = transforms.Compose([
		transforms.ToTensor(),
		normalize
		])
      
		trainSet = torchvision.datasets.ImageNet(root='./ImageNet', split='train', download=False, transform=transforms_train)	#set download=False because ImageNet is not longer accessible, and need to download manually
      
		testSet = torchvision.datasets.ImageNet(root='./ImageNet', split='val', download=False, transform=transforms_test)
      
		tranloader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=32)
      
		testloader = torch.utils.data.DataLoader(testSet, batch_size=256, shuffle=False, num_workers=256)
      
		return tranloader, testloader
