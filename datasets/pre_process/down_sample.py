import cv2
import torch.nn as nn


def down_sample(x, scale_factor=2, kernel_size=(9,9), sigma=3):
	out = cv2.GaussianBlur(x, ksize=kernel_size, sigmaX=sigma,sigmaY=sigma)
	out = cv2.resize(out, (0,0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_CUBIC)
	return out

def down_sample_torch(x, ratio=0.5):
	dim = x.dim()
	out = x if dim == 4 else x.unsqueeze(dim=0)
	out = nn.functional.interpolate(
		out, 
		scale_factor=(ratio, ratio), 
		mode='bicubic', 
		recompute_scale_factor=True, 
		align_corners=True)
	out = out if dim == 4 else out.squeeze(dim=0)
	return out


if __name__=='__main__':
	import numpy as np
	import torch 

	data_path = '/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy'
	hsi = np.load(data_path)
	print('hr shape:', hsi.shape)
	print('hr dtype:', hsi.dtype)
	print('hr max:', hsi.max())
	print('hr min:', hsi.min())

	hsi_torch = torch.Tensor(hsi[0].transpose(2,0,1))

	lr = down_sample_torch(hsi_torch, ratio=1/4).cpu().numpy().transpose(1,2,0)

	print('lr shape:', lr.shape)
	print('lr dtype:', lr.dtype)
	print('lr max:', lr.max())
	print('lr min:', lr.min())
	
	cv2.imwrite('hr.png', np.mean(hsi[0], axis=2)*255)
	cv2.imwrite('lr.png', np.mean(lr, axis=2)*255)
