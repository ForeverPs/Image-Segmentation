import os
import cv2
import numpy as np
import nibabel as nib
from scipy import misc

class Convert(object):
	def __init__(self):
		print('\n' + '-' * 30 + '\n' + ' ' * 7 +'INITIALIZATION\n' + '-' * 30)
		self.train = 'train_temp/image/'
		self.label = 'train_temp/groundtruth/'
		self.test = 'test_temp/image/'
		self.images = os.listdir(self.train)
		self.images.sort(key=lambda x:int(x[4:-4]))
		self.labels = os.listdir(self.label)
		self.labels.sort(key=lambda x:int(x[9:-4]))
		self.test_images = os.listdir(self.test)
		self.test_images.sort(key=lambda x:int(x[4:-4]))
		assert len(self.images) == len(self.labels), 'Mismatch'
		self.num_train = len(self.images)
		self.num_test = len(self.test_images)

	def nii2tif(self, test=True, train=True, label=True):
		print('\n\nConvert .nii to .tif')
		print(30 * '-')
		if train:
			if not os.path.lexists('data/train/image'):
				os.makedirs('data/train/image')
		if label:
			if not os.path.lexists('data/train/label'):
				os.makedirs('data/train/label')
		if test:
			if not os.path.lexists('data/test'):
				os.makedirs('data/test')
		print('Wait...')
		if train:
			for ele in self.images:
				path = self.train + ele
				self.get_nii(path, prefix='data/train/image/')
		if test:
			for ele in self.test_images:
				path = self.test + ele
				self.get_nii(path, prefix='data/test/')
		if label:
			for ele in self.labels:
				path = self.label + ele
				self.get_nii(path, prefix='data/train/label/')
		print('Finished.\n' + 30 * '-')

	def get_nii(self, path, prefix):
		img = nib.load(path).get_data()
		for i in range(np.shape(img)[-1]):
			name = path[path.rindex('/')+1:path.rindex('.nii')] + str(i)
			self.store(img[:,:,i], prefix + name)

	def store(self, img, name, tar='.tif'):
		misc.imsave(name + tar, img)

if __name__ == '__main__':
	c = Convert()
	c.nii2tif()
