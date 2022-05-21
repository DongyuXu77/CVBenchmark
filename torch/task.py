class trial:
	def __init__(self):
		self.metrics = self.get_metrics()
		self.model = self.get_model()
		self.optimizer = self.get_optimizer()
		self.lr_schedule = self.get_lr_schedulr()
		self.criterion = self.get_criterion()
        self.device = 

	def train():
        self.model.train()
		for epoch in range (epochs):
			self.initialize_metrics()
			for batch, (input, label) in enumerate self.dataloader['Train']:
				self.train_step(epoch, batch, input, label)
			self.lr_schedule.step()

    def train_step(epoch: int, batch: int):
        input, label = input.to(self.device), label.to(self.device)
        output = self.model.forward
        if (batch+1)%batch_log==0:
            self.log()

    def log(self):
        for key, value in self.metrics:
            print(f"{key} : {value}")
