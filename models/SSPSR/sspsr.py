import torch
import math
import torch.nn as nn
from .common import *
from ..CANet.fusion import *


class SSB(nn.Module):
		def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv, fus_mode=None):
				super(SSB, self).__init__()
				self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
				self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

				if fus_mode=='concat':
					self.fus = FusConcatLayer(n_feats, n_feats)
				elif fus_mode=='affine':
					self.fus = FusAffineLayer(n_feats, n_feats)
				elif fus_mode==None:
					self.fus = None
				else:
					raise Exception('Unknown fus_mode')

				print('[SSB] fus_mode:', fus_mode)

		def forward(self, x0):
				x, feat_cluster = x0

				out = self.spa(x)

				if self.fus is not None:
					out, _ = self.fus((out, feat_cluster))

				out = self.spc(out)
				return out, feat_cluster


class SSPN(nn.Module):
		def __init__(self, n_feats, n_blocks, act, res_scale, fus_mode=None):
				super(SSPN, self).__init__()

				kernel_size = 3
				m = []

				for i in range(n_blocks):
						m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale, fus_mode=fus_mode))

				self.net = nn.Sequential(*m)

		def forward(self, x0):
				x, feat_cluster = x0
				res, _ = self.net((x, feat_cluster))
				res += x

				return res


# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
		def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv, fus_mode=None):
				super(BranchUnit, self).__init__()
				kernel_size = 3
				self.head = conv(n_colors, n_feats, kernel_size)
				self.body = SSPN(n_feats, n_blocks, act, res_scale, fus_mode=fus_mode)
				self.upsample = Upsampler(conv, up_scale, n_feats)
				self.tail = None

				if use_tail:
						self.tail = conv(n_feats, n_colors, kernel_size)

		def forward(self, x0):
				x, feat_cluster = x0
				y = self.head(x)
				y = self.body((y, feat_cluster))
				y = self.upsample(y)
				if self.tail is not None:
						y = self.tail(y)

				return y


class SSPSR(nn.Module):
		def __init__(self, channels, scale_factor, n_clusters, fus_mode='affine'):
				super(SSPSR, self).__init__()

				self.n_clusters = n_clusters

				n_colors = channels
				n_scale = scale_factor

				n_subs = 8
				n_ovls = 2
				n_blocks = 3
				n_feats = 256
				res_scale = 0.1
				conv = default_conv

				kernel_size = 3
				self.shared = True
				
				act = nn.ReLU(True)

				# calculate the group number (the number of branch networks)
				self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
				# calculate group indices
				self.start_idx = []
				self.end_idx = []

				# upsample bicubic
				self.up = nn.Upsample(scale_factor=n_scale, mode='bicubic', align_corners=True)

				for g in range(self.G):
						sta_ind = (n_subs - n_ovls) * g
						end_ind = sta_ind + n_subs
						if end_ind > n_colors:
								end_ind = n_colors
								sta_ind = n_colors - n_subs
						self.start_idx.append(sta_ind)
						self.end_idx.append(end_ind)

				if self.shared:
						self.branch = BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, up_scale=n_scale//2, conv=default_conv, fus_mode=fus_mode)
						# up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
				else:
						self.branch = nn.ModuleList()
						for i in range(self.G):
								self.branch.append(BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, up_scale=2, conv=default_conv, fus_mode=fus_mode))

				self.trunk = BranchUnit(n_colors, n_feats, n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv, fus_mode=None)		# trunk not fuse cluster map
				self.skip_conv = conv(n_colors, n_feats, kernel_size)
				self.final = conv(n_feats, n_colors, kernel_size)
				self.sca = n_scale//2

				self.conv_cluster1 = nn.Sequential(
					conv(n_clusters, n_feats, kernel_size=3),
					conv(n_feats, n_feats, kernel_size=3),
				)
				# self.conv_cluster2 = SSB(n_feats, kernel_size=3,act=act, res_scale=1, conv=default_conv, fus_mode=None)

		def forward(self, x0):
				x, cluster = x0
				
				# cluster features
				cluster = nn.functional.one_hot(cluster, num_classes=self.n_clusters).to(x.device).to(torch.float32)
				cluster = cluster.permute(0,3,1,2)
				feat_cluster = self.conv_cluster1(cluster)
				# feat_cluster, _ = self.conv_cluster2((feat_cluster, None))

				# feat_cluster = self.conv_cluster1(x)

				b, c, h, w = x.shape

				lms = self.up(x)

				# Initialize intermediate “result”, which is upsampled with n_scale//2 times
				y = torch.zeros(b, c, self.sca * h, self.sca * w).to(x.device)

				channel_counter = torch.zeros(c).to(x.device)

				for g in range(self.G):
						sta_ind = self.start_idx[g]
						end_ind = self.end_idx[g]

						xi = x[:, sta_ind:end_ind, :, :]
						if self.shared:
								xi = self.branch((xi, feat_cluster))
						else:
								xi = self.branch[g]((xi, feat_cluster))

						y[:, sta_ind:end_ind, :, :] += xi
						channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

				# intermediate “result” is averaged according to their spectral indices
				y = y / channel_counter.unsqueeze(1).unsqueeze(2)

				y = self.trunk((y, feat_cluster))
				y = y + self.skip_conv(lms)
				y = self.final(y)

				return y