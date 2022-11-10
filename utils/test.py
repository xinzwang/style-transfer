"""
Test api
"""
import os
import cv2
import numpy as np
import imgvision as iv
from tqdm import tqdm

import torch

def test(model, dataloader, loss_fn, device):
	total_loss, total_content_loss, total_style_loss, total_tv_loss = [], [], [], []
	for i, (img, style_img) in enumerate(tqdm(dataloader)):
		img = img.to(device)
		style_img = style_img.to(device)
		with torch.no_grad():
			pred = model(img)
		assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		loss, loss_content, loss_style, loss_tv = loss_fn(pred, img, style_img)
		total_loss.append(loss.item())
		total_content_loss.append(loss_content.item())
		total_style_loss.append(loss_style.item())
		total_tv_loss.append(loss_tv.item())
		
	total_loss = np.mean(total_loss)
	total_content_loss = np.mean(total_content_loss)
	total_style_loss = np.mean(total_style_loss)
	total_tv_loss = np.mean(total_tv_loss)
	print('[TEST] loss:%.5f content:%.5f style:%.5f tv:%.5f' % (total_loss, total_content_loss, total_style_loss, total_tv_loss))
	return total_loss, total_content_loss, total_style_loss, total_tv_loss

def visual(model, dataloader, img_num=3, save_path='img/', err_gain=10, device=None):
	# create save dir
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# infer and save
	it = iter(dataloader)
	for i in range(min(img_num, dataloader.__len__())):
		img, _ = next(it)
		img = img.to(device)
		with torch.no_grad():
			pred = model(img)
		assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		# torch->numpy; 1CHW->HWC; [0, 1]; BGR
		pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
		# save image
		cv2.imwrite(save_path + '%d.png'%(i), pred_*255.0)
	return