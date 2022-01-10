import torch
import torchvision
from torchvision import datasets, transforms

def dataloader(config):
    if config.dataset == 'ImageNet':
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
        trainSet = torchvision.datasets.ImageNet(root='./ImageNet_data', train=True, download=True, transform=transforms_train)
        testSet = torchvision.datasets.ImageNet(root='./ImageNet_data', train=False, download=True, transform=transforms_test)
        tranloader = torch.utils.data.DataLoader(trainSet, batch_size=config.trainBatchsize, shuffle=True, num_workers=config.workers)
        testloader = torch.utils.data.DataLoader(testSet, batch_size=config.testBatchsize, shuffle=False, num_workers=config.workers)
        return tranloader, testloader