import torch

class trial:
	def __init__(self, configDict):
		self.metrics = self.get_metrics(configDict['metircs'])
		self.model = self.get_model(configDict['model'])
		self.optimizer = self.get_optimizer(configDict['optimzier'])
		self.lr_schedule = self.get_lr_schedule(configDict['lr_shcedule'])
		self.criterion = self.get_criterion(configDict['criterion'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def train():
        self.model.train()
		for epoch in range (epochs):
			self.initialize_metrics()
			for batch, (data, label) in enumerate self.dataloader['Train']:
				self.single_step(data=data, label=label, epoch=epoch, batch=batch)
			self.lr_schedule.step()

    def single_step(mode: str='Train',  data, label, **kwargs):
        data, label = data.to(self.device), label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, label)
        if mode=='Train':
            loss.backward()
            self.optimizer.step()
        if mode=='Train' and (kwargs['batch']+1)%batch_log==0:
            self.log(kwargs)

    def log(self,mode: str='Train', ** kwargs):
        if mode=='Train':
            print(f"[Epoch:{kwargs['epcoh']+1}  Batch:{kwargs['batch']+1}]")
            print('_'*100)
        for key, value in self.metrics:
            print(f"{key} : {value}")


    def test():
        self.model.eval()
        self.inititalize_metrics()
        for _, (data, label) in enumerate self.dataloader['Test']:
            self.single_step(mode='Test')
        self.log()
            
