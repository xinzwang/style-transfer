import torch

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return