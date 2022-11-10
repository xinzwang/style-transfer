import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from .pre_process.general import letterbox

class COCODataset(Dataset):
	def __init__(self, path, style_img, img_size=(224, 224), test_flag=False):
		super().__init__()
		self.img_size=img_size
		self.img_paths = glob.glob(path + ('images/val2017/' if test_flag else 'images/val2017/') + '*.jpg')[0:500]
		if test_flag:
			self.img_paths = self.img_paths[0:10]
		# self.img_paths = glob.glob(path)[0:50]
		self.style_img = cv2.imread(style_img).astype(np.float32) / 255.0
		self.style_img_resized = cv2.resize(self.style_img, img_size)
		# self.style_shape = self.style_img.shape
		self.test_flag = test_flag
		return

	def __len__(self):
		return len(self.img_paths)
	
	def __getitem__(self, index):
		p = self.img_paths[index]
		img = cv2.imread(p).astype(np.float32) / 255.0	# np.uint8->np.float32; [0, 255]->[0, 1]
		if self.test_flag:
			H,W,C = img.shape
			img = img[:H-H%4, :W-W%4, :]
			style_img = cv2.resize(self.style_img, (img.shape[1], img.shape[0]))
			return img.transpose(2, 0, 1), style_img.transpose(2, 0, 1)
		else:
			img, _, _ = letterbox(img, new_shape=self.img_size,  auto=False, scaleFill=True)
		# x = np.random.randint(0, self.style_shape[0]-self.img_size[0])
		# y = np.random.randint(0, self.style_shape[1]-self.img_size[1])
		# style_img = self.style_img[x:x+self.img_size[0], y:y+self.img_size[1], :]
		# cv2.imwrite('img.png', self.style_img*255)
		# noise = np.random.normal(0.2, 1, size=(self.img_size[0], self.img_size[1], 3))
		# img +=noise
		return img.transpose(2, 0, 1), self.style_img_resized.transpose(2, 0, 1)


