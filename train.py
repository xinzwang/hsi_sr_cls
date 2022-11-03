import os
import cv2
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.logger import create_logger
from utils.dataset import build_dataset
from utils.test import test, visual
from utils.seed import set_seed
from utils.core import SRClusterCore

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE', choices=['ICVL', 'CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian'])
	parser.add_argument('--scale_factor', default=2, type=int)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--epoch', default=10001)
	parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
	parser.add_argument('--seed', default=17, type=int)
	parser.add_argument('--device', default='cuda:2')
	parser.add_argument('--parallel', default=False)
	parser.add_argument('--device_ids', default=['cuda:5', 'cuda:6', 'cuda:7'])
	parser.add_argument('--model', default='SSPSR')
	parser.add_argument('--cls_num', default=19, type=int)
	parser.add_argument('--fus_mode', default='affine')
	# Pavia
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy')
	# parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256_kmeans_9_0.npy')
	# parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256_kmeans_9_0.npy')
	# parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256_kmeans_hr_9_0.npy')
	# parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256_kmeans_hr_9_0.npy')
	# Salinas
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128.npy')
	# CAVE
	parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train.npy')
	parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test.npy')
	# parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train_kmeans_19_0.npy')
	# parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test_kmeans_19_0.npy')
	parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train_kmeans_hr_19_0.npy')
	parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test_kmeans_hr_19_0.npy')
	# ICVL
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/ICVL/train/')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/ICVL/test/')

	args = parser.parse_args()
	print(args)
	return args

def train(args):
	t = time.strftime('%Y-%m-%d_%H:%M:%S')
	checkpoint_path = 'checkpoints/%s/%s/%s/' % (args.dataset, args.model, t)
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	log_path = 'log/%s/%s/' %(args.dataset, args.model)
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	logger = create_logger(log_path + '%s.log'%(t))
	logger.info(str(args))

	writer = SummaryWriter('tensorboard/%s/%s/%s/' % (args.dataset, args.model, t))

	# set seed
	set_seed(args.seed)

	# device
	cudnn.benchmark = True
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# dataset
	dataset, dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.train_path,
		cls_path=args.train_cls_path,
		cls_num=args.cls_num,
		batch_size=args.batch_size, 
		scale_factor=args.scale_factor, 
		test_flag=False)
	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.test_path, 
		cls_path=args.test_cls_path,
		cls_num=args.cls_num,
		batch_size=1, 
		scale_factor=args.scale_factor, 
		test_flag=True)

	# core
	core = SRClusterCore(batch_log=10)
	core.inject_logger(logger)
	core.inject_writer(writer)
	core.inject_device(device)
	core.build_model(name=args.model, channels=dataset.channels, scale_factor=args.scale_factor, n_clusters=args.cls_num, fus_mode=args.fus_mode)
	
	# loss optimizer
	loss_fn = nn.L1Loss()
	optimizer = optim.Adam(core.model.parameters(), args.lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50, threshold=1e-4, min_lr=1e-5)
	core.inject_loss_fn(loss_fn)
	core.inject_optim(optimizer)
	core.inject_scheduler(scheduler)


	if args.parallel:
		core.parallel(device_ids = args.device_ids)

	# train loop
	for epoch in range(args.epoch):
		mean_loss = core.train(dataloader)
		logger.info('[TEST] epoch:%d mean_loss:%.5f'%(epoch, mean_loss))

		if epoch % 5 == 0:
			psnr, ssim, sam, mse = test(core.model, test_dataloader, device=device)
			logger.info('[TEST] epoch:%d psnr:%.5f ssim:%.5f sam:%.5f mse:%.5f' %(epoch, psnr, ssim, sam, mse))
			writer.add_scalar(tag='score/PSNR', scalar_value=psnr, global_step=epoch)
			writer.add_scalar(tag='score/SSIM', scalar_value=ssim, global_step=epoch)
			writer.add_scalar(tag='score/SAM', scalar_value=sam, global_step=epoch)
			writer.add_scalar(tag='score/MSE', scalar_value=mse, global_step=epoch)

			save_path = checkpoint_path + 'epoch=%d_psnr=%.5f_ssim=%.5f'%(epoch, psnr,ssim)
			visual(core.model, test_dataloader, img_num=3, save_path=save_path + '/', device=device)
			core.save_ckpt(save_path +'ckpt.pt', dataset=args.dataset)
			
		pass
	# save the final model
	save_path = checkpoint_path + 'final_epoch=%d_psnr=%.5f_ssim=%.5f/'%(epoch, psnr,ssim)
	visual(core.model, test_dataloader, img_num=3, save_path=save_path+'img/')
	core.save_ckpt(save_path +'.pt', dataset=args.dataset)
	return


if __name__=='__main__':
	args = parse_args()
	train(args)