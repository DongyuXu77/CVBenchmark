import time
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
	start_time = time.time()
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
		if torch.cuda.current_device()==0 and batch%500==0:
			print("[Epoch{} batch:{}] Accuracy:{:.4f} avg_Loss:{:.4f}".format(epoch+1, batch+1, correct/total, trainLoss/(batch+1)))
	if torch.cuda.current_device()==0:
		print("[Epoch:{}] Accuracy:{:.4f} Duration:{:.1f}s".format(epoch+1, correct/total, time.time()-start_time))

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
            		print("[Epoch:{}] Accuracy:{:.4f}".format(epoch+1, correct/total))

def test(model, epoch, testLoader, device):
	correct = 0
	total = 0
	model.eval()
	start_time = time.time()
	with torch.no_grad():
		for batch, (inputs, labels) in enumerate(testLoader):
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, predict = outputs.max(1)
			correct = correct+predict.eq(labels).sum().item()
			total = total+inputs.size(0)
		if torch.cuda.current_device()==0:
			print("[Epoch:{}] Accuracy:{:.4f} Duration:{:.1f}s".format(epoch+1, correct/total, time.time()-start_time))
		if correct/total>best_acc:
			best_acc = correct/total
			save(device=device)

def save(path='./save_weights/best_ckpt.pth', device='cuda'):
	if os.path.exists(os.path.dirname((path))==False:
		os.makedirs(os.path.dirname(path))
	model = model if device=='cpu' else model.module
	torch.save(mode.state_dict(), path)

def load(path='./save_weights/best_ckpt.pth', device='cuda'):
	if os.path.exists(path):
		weights_dict = torch.load(path)
		model = model if device=='cpu' else model.module
		model.load_state_dict(weights_dict)
	else:
		raise ValError('weightFile does not exists')

if __name__=="__main__":
	model = VGG()
	model, device, is_distributed = gpu_set(model)
	trainsampler, trainloader, testsampler, testloader = dataloader({'dataset': "ImageNet", 'is_distributed':is_distributed})
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	best_acc = 0
	for epoch in range(200):
		if is_distributed:
			trainsampler.set_epoch(epoch)
			testsampler.set_epoch(epoch)
		train(model, epoch, trainloader, optimizer, criterion, device)
		# eval(model, epoch, evalloader, optimizer, device)
		test(model, epoch, testloader, device)
