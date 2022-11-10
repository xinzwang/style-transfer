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
from utils.core import StyleTransCore

from models.loss import PerceptualLoss

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='COCO', choices=['COCO'])
	parser.add_argument('--img_size', default=256, type=int)
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--epoch', default=10001)
	parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate')
	parser.add_argument('--seed', default=17, type=int)
	parser.add_argument('--device', default='cuda:4')
	parser.add_argument('--parallel', default=False)
	parser.add_argument('--device_ids', default=['cuda:5', 'cuda:6', 'cuda:7'])
	parser.add_argument('--model', default='StyleNet')
	parser.add_argument('--data_path', default='/share/dataset/coco/')
	# parser.add_argument('--style_img', default='img/style/Classification.png')
	# parser.add_argument('--style_img', default='img/style/The Starry Night.jpg')
	parser.add_argument('--style_img', default='img/style/Sunrise.jpg')
	# parser.add_argument('--style_img', default='img/style/神奈川冲浪里.jpg')
	# parser.add_argument('--style_img', default='img/style/ASCII art.png')
	
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
		path=args.data_path,
		style_img = args.style_img,
		img_size=(args.img_size, args.img_size),
		batch_size=args.batch_size, 
		test_flag=False)
	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.data_path,
		style_img = args.style_img,
		img_size=(args.img_size, args.img_size),
		batch_size=1,
		test_flag=True)

	# core
	core = StyleTransCore(batch_log=10)
	core.inject_logger(logger)
	core.inject_writer(writer)
	core.inject_device(device)
	core.build_model(name=args.model)
	
	# loss optimizer
	# loss_fn = PerceptualLoss(layer_indexs=(3, 8, 15, 32), device=device)
	loss_fn = PerceptualLoss(device=device)
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
		logger.info('[TRAIN] epoch:%d mean_loss:%.5f'%(epoch, mean_loss))

		# test set
		loss, loss_content, loss_style, loss_tv = test(core.model, test_dataloader, loss_fn, device=device)
		logger.info('[TEST] loss:%.5f content:%.5f style:%.5f tv:%.5f' % (loss, loss_content, loss_style, loss_tv))
		writer.add_scalar(tag='test/loss', scalar_value=loss, global_step=epoch)
		writer.add_scalar(tag='test/loss_content', scalar_value=loss_content, global_step=epoch)
		writer.add_scalar(tag='test/loss_style', scalar_value=loss_style, global_step=epoch)
		writer.add_scalar(tag='test/loss_tv', scalar_value=loss_tv, global_step=epoch)

		save_path = checkpoint_path + 'epoch=%d'%(epoch)
		visual(core.model, test_dataloader, img_num=15, save_path=save_path + '/', device=device)
		core.save_ckpt(save_path +'ckpt.pt', dataset=args.dataset)
			
		pass
	# save the final model
	save_path = checkpoint_path + 'final_epoch=%d'%(epoch)
	visual(core.model, test_dataloader, img_num=15, save_path=save_path + '/', device=device)
	core.save_ckpt(save_path +'ckpt.pt', dataset=args.dataset)
	return


if __name__=='__main__':
	args = parse_args()
	train(args)