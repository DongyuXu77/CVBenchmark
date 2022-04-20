class trial:
	def __init__(self):
		self.metrics = self.get_metrics()
		self.model = self.get_model()
		self.optimizer = self.get_optimizer()
		self.lr_schedule = self.get_lr_schedulr()
		self.criterion = self.get_criterion()

	def train():
		for epoch in range (epochs):
			self.initialize_metrics()
			for batch, (input, label) in enumerate self.dataloader['Train']:
				self.train_step(epoch, batch)
			self.lr_schedule.step()

	def train_step(epoch, batch):
		self.
