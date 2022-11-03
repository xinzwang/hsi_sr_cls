"""
Test api
"""
import os
import cv2
import numpy as np
import imgvision as iv
from tqdm import tqdm

import torch

def test(model, dataloader, device):
	psnr, ssim, sam, mse = [], [], [], []
	for i, (lr, hr, cluster) in enumerate(tqdm(dataloader)):
		lr = lr.to(device)
		cluster = cluster.to(device)
		with torch.no_grad():
			pred = model((lr,cluster))
		assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		# torch->numpy; 1CHW->HWC; [0, 1]
		hr_ = hr.cpu().numpy()[0].transpose(1,2,0)
		pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
		# eval
		metric = iv.spectra_metric(pred_, hr_, max_v=1.0)
		psnr_, ssim_, sam_, mse_ = metric.PSNR(), metric.SSIM(), metric.SAM(), metric.MSE()
		psnr.append(psnr_)
		ssim.append(ssim_)
		sam.append(sam_)
		mse.append(mse_)
	psnr, ssim, sam, mse = np.mean(psnr),np.mean(ssim),np.mean(sam),np.mean(mse)
	print('[TEST] psnr:%.5f ssim:%.5f sam:%.5f mse:%.5f' %(psnr, ssim, sam, mse))
	return psnr, ssim, sam, mse

def visual(model, dataloader, img_num=3, save_path='img/', err_gain=10, device=None):
	# create save dir
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# infer and save
	it = iter(dataloader)
	for i in range(min(img_num, dataloader.__len__())):
		lr, hr, cluster = next(it)
		lr = lr.to(device)
		cluster = cluster.to(device)
		with torch.no_grad():
			pred = model((lr, cluster))
		assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		# torch->numpy; 1CHW->HWC; [0, 1]
		hr_ = hr.cpu().numpy()[0].transpose(1,2,0)
		pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
		# save err_map gray
		err = np.mean(np.abs(pred_ - hr_), axis=2)
		gray = np.mean(pred_, axis=2)
		cv2.imwrite(save_path + '%d_err.png' %(i), cv2.applyColorMap((err * 255 * err_gain).astype(np.uint8), cv2.COLORMAP_JET))
		cv2.imwrite(save_path + '%d_gray.png'%(i), gray * 255.0)
	return