import torch
import torchvision
from torchvision import datasets, transforms

def dataloader(config):
	if config['dataset'] == 'ImageNet':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])
		transforms_train = transforms.Compose([
		transforms.RandomCrop(224, pad_if_needed=True),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize
		])
		transforms_test = transforms.Compose([
		transforms.RandomCrop(224, pad_if_needed=True),
		transforms.ToTensor(),
		normalize
		])
		trainSet = torchvision.datasets.ImageNet(root='./ILSVRC2012', split='train', download=False, transform=transforms_train)	#set download=False because ImageNet is not longer accessible, and need to download manually
		testSet = torchvision.datasets.ImageNet(root='./ILSVRC2012', split='val', download=False, transform=transforms_test)
	trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet) if config['is_distributed'] else None
	testSampler = torch.utils.data.distributed.DistributedSampler(testSet) if config['is_distributed'] else None
	tranloader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=(trainSample is None), num_workers=32, sampler=trainSampler)
	testloader = torch.utils.data.DataLoader(testSet, batch_size=256, shuffle=(testSample is None), num_workers=256, sampler=testSampler)
	return trainsampler, tranloader, testsampler, testloader
