"""
Core for training assembly
"""
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import models

class SRClusterCore:
	def __init__(self, batch_log=10):
		self.parallel = False
		self.batch_log = batch_log
		self.batch_cnt = 0
		self.epoch_cnt = 0
		pass

	def inject_logger(self, logger):
		self.logger = logger

	def inject_writer(self, writer):
		self.writer = writer

	def inject_device(self, device):
		self.device=device

	def build_model(self, name, channels=3, scale_factor=2, n_clusters=9, fus_mode='affine'):
		self.model = getattr(models, name)(channels, scale_factor, n_clusters=n_clusters, fus_mode=fus_mode).to(self.device)
	
	def build_loss(self):
		self.loss_fn = getattr(self.model.loss_name, )

	def inject_loss_fn(self, loss_fn):
		self.loss_fn = loss_fn

	def inject_optim(self, optimizer):
		self.optimizer = optimizer

	def inject_scheduler(self, scheduler):
		self.scheduler = scheduler
	
	def parallel(self, device_ids=['cuda:0']):
		if self.parallel==False:
			self.model = nn.DataParallel(self.model, device_ids=device_ids)
			self.optimizer = nn.DataParallel(self.optimizer, device_ids=args.device_ids)
			self.scheduler = nn.DataParallel(self.scheduler, device_ids=args.device_ids)

	def train(self, dataloader):
		if self.parallel==True:
			mean_loss = self.train_parallel(dataloader)
		else:
			mean_loss = self.train_single(dataloader)
		return mean_loss

	def train_single(self, dataloader):
		logger = self.logger
		writer = self.writer
		device = self.device
		model = self.model
		loss_fn = self.loss_fn
		optimizer = self.optimizer
		scheduler = self.scheduler

		self.epoch_cnt += 1

		total_loss = []
		c_lr = optimizer.state_dict()['param_groups'][0]['lr']
		logger.info('  lr:%f'%(c_lr))
		writer.add_scalar(tag='train/lr', scalar_value=c_lr, global_step=self.epoch_cnt)

		for i, (lr, hr, cluster) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)
			cluster = cluster.to(device)

			optimizer.zero_grad()
			pred = model((lr, cluster))
			loss = loss_fn(pred, hr)
			loss.backward()
			optimizer.step()

			total_loss.append(loss.item())
			self.batch_cnt += 1
			if i % self.batch_log == 1:
				logger.info('  batch:%d loss:%.5f' % (i, loss.item()))
				writer.add_scalar(tag='train/loss', scalar_value=loss.item(), global_step=self.batch_cnt)
			pass
		mean_loss = np.mean(total_loss)
		scheduler.step(mean_loss)
		return mean_loss

	def train_parallel(self, dataloader):
		logger = self.logger
		device = self.device
		model = self.model
		loss_fn = self.loss_fn
		optimizer = self.optimizer
		scheduler = self.scheduler

		total_loss = []
		c_lr = optimizer.module.state_dict()['param_groups'][0]['lr']
		logger.info('  lr:%f'%(c_lr))

		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)

			optimizer.zero_grad()
			pred = model(lr)
			loss = loss_fn(pred, hr)
			loss.backward()
			optimizer.module.step()

			total_loss.append(loss.item())
			if i % self.batch_log == 1:
				logger.info('  batch:%d loss:%.5f' % (i, loss.item()))
			pass
		mean_loss = np.mean(total_loss)
		scheduler.module.step(mean_loss)
		return mean_loss

	def save_ckpt(self, save_path, dataset):
		torch.save({
			'dataset': dataset,
			'model': self.model
		}, save_path)
		return
		
