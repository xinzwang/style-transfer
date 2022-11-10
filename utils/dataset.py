"""
Build Dataset and dataloader
"""

from torch.utils.data.dataloader import DataLoader
from datasets import *

def SelectDatasetObject(name):
	if name in ['COCO']:
		return COCODataset
	else:
		raise Exception('Unknown dataset:', name)

def build_dataset(dataset, path, style_img, img_size, batch_size=32, scale_factor=2, test_flag=False):
	datasetObj = SelectDatasetObject(dataset)
	dataset = datasetObj(
		path=path,
		style_img=style_img,
		img_size=img_size,
		test_flag=test_flag,
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size if not test_flag else 1,
		num_workers=8,
		shuffle= (not test_flag)	# shuffle only train
	)
	return dataset, dataloader
