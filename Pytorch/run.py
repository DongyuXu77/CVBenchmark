import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
from model import *

bestAccuracy=[1, 0]
def parserArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help="")
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

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
        print("[Epoch:{} batch:{}] Accuracy:{:.2f} Loss:{:.2f}".format(epoch+1, batch+1, correct/total, loss/(batch+1)))

def eval(model, epoch, evalLoader, optimizer, device):
    global bestAccuracy
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
            print("[Epoch:{} batch:{}] Accuracy:{:.2f}".format(epoch+1, batch+1, correct/total))
    if correct/total<bestAccuracy[0]:
        pass

def test(model, epoch,testLoader,device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

if __name__=="__main__":
    config = parserArgs()
    model = vgg()
    model.to(config.device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in config.epoch:
        train(model, epoch, trainloader, optimizer, criterion, config.device)
        eval(model, epoch, evalloader, optimizer, config.device)
        test(epoch)