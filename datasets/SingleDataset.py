"""
SingleDataset

One file with all training images. 

Data file must be stored in the form of numpy.array, with the shape of [N, H, W, C].
Where N represents the number of image, H and W represent the image width, and C represents the number of channels
"""

import os
import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .pre_process.down_sample import down_sample
from .pre_process.enhance import *


class SingleDataset(Dataset):
	def __init__(self, path, cls_path, cls_num, scale_factor=2, test_flag=False):
		super(SingleDataset, self).__init__()
		self.scale_factor = scale_factor
		self.ratio = 1.0/scale_factor
		self.test_flag = test_flag
		self.cls_num = cls_num

		self.hsi = np.load(path).astype(np.float32)	# NHWC; [0, 1]; np.float32
		self.cluster = np.load(cls_path).astype(np.int64)	# NHW; [0, cls_num); np.int64 

		self.shape = self.hsi.shape
		self.channels = self.shape[-1]
		self.len = self.shape[0]
		return
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		hr = self.hsi[index, ...]
		cluster = self.cluster[index, ...]
		
		# H, W, C = hr.shape
		# if (H % self.scale_factor !=0) or (W % self.scale_factor != 0):
		# 	hr = hr[0:H//self.scale_factor*self.scale_factor, 0:W//self.scale_factor*self.scale_factor, :]
		# 	cluster = cluster[0:H//self.scale_factor*self.scale_factor, 0:W//self.scale_factor*self.scale_factor]

		if self.test_flag:
			pass
		else:
			# enhance
			hr, cluster = rot90(hr, cluster)
			hr, cluster = flip(hr, cluster)
			hr = np.ascontiguousarray(hr)
			cluster = np.ascontiguousarray(cluster)
			pass

		# down sample
		lr = down_sample(hr, scale_factor=self.scale_factor, kernel_size=(9,9), sigma=3)

		return lr.transpose(2,0,1), hr.transpose(2,0,1), cluster


if __name__ == '__main__':
	import glob
	import cv2
	import numpy as np

	paths = glob.glob('/data2/wangxinzhe/codes/datasets/CAVE/hsi/*.npy')
	print(paths)
	for i, x in enumerate(paths):
		gt = np.load(x)
		print('gt shape', gt.shape)
		print('gt min:', gt.min())
		print('gt max:', gt.max())

		cv2.imwrite('img/cave/%s.png'%(x.split('/')[-1].split('.')[0]), np.mean(gt, axis=2)/ gt.max() * 255 )
