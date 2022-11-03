"""
Build Dataset and dataloader
"""

from torch.utils.data.dataloader import DataLoader
from datasets import *

def SelectDatasetObject(name):
	if name in ['Pavia', 'PaviaU', 'Salinas', 'KSC', 'Indian', 'CAVE']:
		return SingleDataset
	elif name in ['ICVL']:
		return MultiDataset
	else:
		raise Exception('Unknown dataset:', name)

def build_dataset(dataset, path, cls_path, cls_num, batch_size=32, scale_factor=2, test_flag=False):
	datasetObj = SelectDatasetObject(dataset)
	dataset = datasetObj(
		path=path,
		cls_path=cls_path,
		cls_num=cls_num,
		scale_factor=scale_factor, 
		test_flag=test_flag,
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size if not test_flag else 1,
		num_workers=8,
		shuffle= (not test_flag)	# shuffle only train
	)
	return dataset, dataloader
