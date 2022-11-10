import cv2

import torch
import torch.nn as nn
from torchvision.models import vgg16

def gram(x):
	"""Gram Matrix
	Gram matrix eval the distans
	"""
	(N, C, H, W) = x.size()
	f = x.view(N, C, H * W)	# cal
	f_T = f.transpose(1, 2)
	G = f.bmm(f_T) / (C * H * W)
	return G


class PerceptualLoss(nn.Module):
	def __init__(self, device=None):
		super(PerceptualLoss, self).__init__()
		self.loss_fn = nn.MSELoss()
		self.device=device

		self.vgg = vgg16(pretrained=True, progress=True).features
		self.vgg.eval()
		for p in self.vgg.parameters():
			p.requires_grad=False
	
	def get_feature_module(self, layer_index):
		res = self.vgg[0:layer_index+1].to(self.device)
		return res

	def vgg_feat(self, x, img, layer_index):
		m = self.get_feature_module(layer_index)
		feat1 = m(x)
		feat2 = m(img)
		return feat1, feat2

	def content_loss(self, img1, img2, layer_indexs):
		loss = 0
		for idx in layer_indexs:
			feat1, feat2 = self.vgg_feat(img1, img2, idx)
			loss += self.loss_fn(feat1, feat2)
		return loss

	def style_loss(self, img1, img2, layer_indexs):
		loss = 0
		for idx in layer_indexs:
			feat1, feat2 = self.vgg_feat(img1, img2, idx)
			gram1 = gram(feat1)
			gram2 = gram(feat2)
			loss += self.loss_fn(gram1, gram2)
		return loss

	def forward(self, pred, img, style_img):
		content_p = 1
		style_p = 5e6
		tv_p =1e-7

		content_loss = content_p * self.content_loss(pred, img, layer_indexs=(15,))								# content loss
		style_loss = style_p * self.style_loss(pred, style_img, layer_indexs=(3, 8, 15, 32))	# style loss

		diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
		diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
		tv_loss = tv_p * (diff_i + diff_j)
		 
		return content_loss + style_loss + tv_loss, content_loss, style_loss, tv_loss