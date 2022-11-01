import cv2
import numpy as np
import torch 

def rot90_torch(img, k=None):
	"""Rotate image for k*pi/2. k=[0,1,2,3], Random rotate if k is None"""
	if k is None:
		k = np.random.randint(0,4)

	dim = img.dim()

	if dim==2:
		dims_=[0,1]
	elif dim==3:
		dims_=[1,2]
	elif dim==4:
		dims_=[2, 3]
	
	img = torch.rot90(img, k=k, dims=dims_)
	return img

def flip_torch(img, k=None):
	"""Flip image horizontally or vertically. k=[0,1,2], Random flip if k is None"""
	if k is None:
		k = np.random.randint(0,3)
	img = img.flip(-k) if k != 0 else img
	return img

def rot90(img, k=None):
	"""Rotate image for k*pi/2. k=[0,1,2,3], Random rotate if k is None"""
	if k is None:
		k = np.random.randint(0,4)
	return np.rot90(img, k)

def flip(img, k=None):
	"""Flip image horizontally or vertically. k=[0,1,2], Random flip if k is None"""
	if k is None:
		k = np.random.randint(0,3)
	
	if k==0:
		return img
	
	return np.flip(img, axis=k)