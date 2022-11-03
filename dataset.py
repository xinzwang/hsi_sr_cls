import os
import sys
import cv2
import numpy as np
import torch
import argparse

from utils.dataset import build_dataset


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE', choices=['ICVL', 'CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian'])
	parser.add_argument('--scale_factor', default=2, type=int)
	parser.add_argument('--batch_size', default=9, type=int)
	parser.add_argument('--cls_num', default=19, type=int)
	# Pavia
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy')
	# parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256_kmeans_9_0.npy')
	# parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256_kmeans_9_0.npy')
	# Salinas
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128.npy')
	# parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128_kmeans_16_0.npy')
	# parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128_kmeans_16_0.npy')
	# CAVE
	parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train.npy')
	parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test.npy')
	parser.add_argument('--train_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train_kmeans_19_0.npy')
	parser.add_argument('--test_cls_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test_kmeans_19_0.npy')
	# ICVL
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/ICVL/train/')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/ICVL/test/')

	args = parser.parse_args()
	print(args)
	return args

def display(data):
	lr, hr, cluster = data
	lr = lr.cpu().numpy().transpose(0,2,3,1)
	hr = hr.cpu().numpy().transpose(0,2,3,1)
	cluster = hr.cpu().numpy().transpose(0,2,3,1)
	num = len(lr)
	plt.figure(figsize=(4*3,3*num), dpi=200)
	for i, x in enumerate(zip(lr,hr,cluster)):
		plt.subplot(1,num,i+1)
		plt.imshow(np.mean(x, axis=2))
	plt.show()

def run(args):

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
		batch_size=args.batch_size, 
		scale_factor=args.scale_factor, 
		test_flag=True)
	
	train_data = next(iter(dataloader))
	test_data = next(iter(test_dataloader))
	

	# train data
	lr, hr, cluster = train_data
	lr = lr.cpu().numpy().transpose(0,2,3,1)
	hr = hr.cpu().numpy().transpose(0,2,3,1)
	cluster = cluster.cpu().numpy()


	path = 'img/%s/train/'%(args.dataset)
	if not os.path.exists(path):
		os.makedirs(path)

	for i, x in enumerate(lr):
		cv2.imwrite(path + '%d_lr.png'%(i), np.mean(x, axis=2) *255)
	for i, x in enumerate(hr):
		cv2.imwrite(path + '%d_hr.png'%(i), np.mean(x, axis=2)*255)
	for i, x in enumerate(cluster):
		img = cv2.applyColorMap((x / args.cls_num * 255).astype(np.uint8), cv2.COLORMAP_JET)
		cv2.imwrite(path + '%d_cls.png'%(i), img)

	# test data
	lr, hr, cluster= test_data
	lr = lr.cpu().numpy().transpose(0,2,3,1)
	hr = hr.cpu().numpy().transpose(0,2,3,1)
	cluster = cluster.cpu().numpy()

	path = 'img/%s/test/'%(args.dataset)
	if not os.path.exists(path):
		os.makedirs(path)

	for i, x in enumerate(lr):
		cv2.imwrite(path + '%d_lr.png'%(i), np.mean(x, axis=2) *255)
	for i, x in enumerate(hr):
		cv2.imwrite(path + '%d_hr.png'%(i), np.mean(x, axis=2)*255)
	for i, x in enumerate(cluster):
		img = cv2.applyColorMap((x / args.cls_num * 255).astype(np.uint8), cv2.COLORMAP_JET)
		cv2.imwrite(path + '%d_cls.png'%(i), img)



if __name__=='__main__':
	args = parse_args()
	run(args)