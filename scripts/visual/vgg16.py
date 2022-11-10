import cv2
import argparse
import os
import numpy as np

import torch
from torchvision.models import vgg16

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_size', default=224, type=int)
	parser.add_argument('--img_name', default='The Starry Night')
	parser.add_argument('--style_img', default='/data2/wangxinzhe/codes/github.com/xinzwang.cn/style-transform/img/style/The Starry Night2.jpg')
	parser.add_argument('--device', default='cuda:0')
	args = parser.parse_args()
	print(args)
	return args


def run(args):
	save_path = 'img/%s/' % (args.img_name)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	img = cv2.imread(args.style_img)
	img = cv2.resize(img, (args.img_size, args.img_size)).transpose(2,0,1)/255
	img = torch.Tensor(img).unsqueeze(0)

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# vgg net
	vgg = vgg16(pretrained=True, progress=True).features
	vgg.eval()
	for p in vgg.parameters():
		p.requires_grad=False

	# run
	layers = (3, 8,15,32)
	for idx in layers:
		img = img.to(device)
		model = vgg[0:idx+1].to(device)
		feat = model(img)

		feat = feat.cpu().squeeze(0).numpy()*255
		print(feat.shape)
		for j, f in enumerate(feat): 
			cv2.imwrite(save_path + 'feat_%d_%d.png'%(idx, j), f)
	

	


if __name__=='__main__':
	args = parse_args()
	run(args)