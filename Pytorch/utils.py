import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def gpu_set(model):
	if torch.cuda.is_available():
		torch.distributed.init_process_group(backend='nccl')
		local_rank = torch.distributed.get_rank()
		torch.cuda.set_device(local_rank)
		device = torch.device("cuda", local_rank)
		model.to(device)
		model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
	return model 
