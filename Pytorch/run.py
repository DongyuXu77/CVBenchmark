import torch
import torchvision
from model import *
import torch.nn as nn
from utils import gpu_set
from data import dataloader
import torch.optim as optim

def train(model, epoch, trainLoader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    trainLoss = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predict = outputs.max(1)
        correct = correct+predict.eq(labels).sum().item()
        total = total+inputs.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        trainLoss = trainLoss + loss.item()
	if torch.cuda.current_device()==0:
        	print("[Epoch:{} batch:{}] Accuracy:{:.4f} Loss:{:.4f}".format(epoch+1, batch+1, correct/total, loss/(batch+1)))

def eval(model, epoch, evalLoader, optimizer, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(evalLoader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predict = outputs.max(1)
            correct = correct + predict.eq(labels).sum().item()
            total = total + inputs.size(0)
	    if torch.cuda.current_device()==0:
            	print("[Epoch:{} batch:{}] Accuracy:{:.4f}".format(epoch+1, batch+1, correct/total))

def test(model, epoch,testLoader,device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predict = outputs.max(1)
            correct = correct+predict.eq(labels).sum().item()
            total = total+inputs.size(0)
	    if toch.cuda.current_device()==0:
            	print("[Epoch:{} batch:{}] Accuracy:{:.4f}".format(epoch+1, batch+1, correct/total))

if __name__=="__main__":
    model = vgg()
    model, device = gpu_set(model)
    tranloader, testloader = dataloader({'dataset': "ImageNet"})
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in config.epoch:
        train(model, epoch, trainloader, optimizer, criterion, device)
        # eval(model, epoch, evalloader, optimizer, device)
        test(model, epoch, testloader, device)
