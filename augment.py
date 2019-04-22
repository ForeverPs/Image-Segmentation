import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import *

class Preparation(object):
	def __init__(self):
		print('\n' + '-' * 30 + '\n' + ' ' * 7 +'INITIALIZATION\n' + '-' * 30)
		self.train = 'data/train/image/'
		self.label = 'data/train/label/'
		self.images = os.listdir(self.train)
		self.images.sort(key=lambda x:int(x[4:-4]))
		self.labels = os.listdir(self.label)
		self.labels.sort(key=lambda x:int(x[9:-4]))
		assert len(self.images) == len(self.labels), 'Mismatch'
		self.number = len(self.images)
		self.generator = ImageDataGenerator(rotation_range=0.2, 
										width_shift_range=0.05, 
										height_shift_range=0.05, 
										shear_range=0.05, 
										zoom_range=0.05, 
										horizontal_flip=True)

	def merge(self):
		print('\n\nMERGE')
		if not os.path.lexists('merge'):
			os.mkdir('merge')
		print('\n' + '-' * 30 + '\nAUGMENTATION\n' + '-' * 30)
		print('Total   Image :', len(self.images))
		for i in range(len(self.images)):
			image = img_to_array(cv2.imread(self.train + self.images[i], cv2.IMREAD_COLOR))
			label = img_to_array(cv2.imread(self.label + self.labels[i], cv2.IMREAD_COLOR))
			image[:,:,-1] = label[:,:,0]
			merged = array_to_img(image)
			merged.save('merge/Case'+str(i+1)+'.tif')
			merged = np.array(merged).reshape((1,) + np.shape(merged))
			savedir = 'augmentation/Case'+str(i+1)
			if not os.path.lexists(savedir):
				os.makedirs(savedir)
			self.augment(merged, savedir, str(i+1))

	def augment(self,img,path,prefix):
		print('\rProcess Image :', int(prefix), end = '')
		gen = self.generator
		i = 0
		for batch in gen.flow(img, batch_size=1, save_to_dir=path,
								save_prefix=prefix, save_format='tif'):
			i += 1
			if i >= 10:# the number of augmentation
				break

	def spilt(self):
		print('\n\n' + '-' * 30 + '\nSPILT\n' + '-' * 30)
		for i in range(len(self.images)):
			print('\rSpilt Image :', i+1, end = '')
			path = 'augmentation/Case'+str(i+1)+'/'
			img_names = os.listdir(path)
			image_folder = 'spilt/train/Case'+str(i+1)
			if not os.path.lexists(image_folder):
				os.makedirs(image_folder)
			label_folder = 'spilt/label/Case'+str(i+1)
			if not os.path.lexists(label_folder):
				os.makedirs(label_folder)
			for name in img_names:
				new = name[name.rindex('_')+1:name.rindex('.tif')]
				img = cv2.imread(path + name, cv2.IMREAD_UNCHANGED)
				img_train = img[:,:,-1]
				img_label = img[:,:,0]
				cv2.imwrite('spilt/train/Case' + str(i+1) + '/' + new + '_train.tif', img_train)
				cv2.imwrite('spilt/label/Case' + str(i+1) + '/' + new + '_label.tif', img_label)


class Process(object):
	def __init__(self, out_rows, out_cols):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.augmentation_path = 'augmentation/'
		self.train_path = 'spilt/train/'
		self.label_path = 'spilt/label/'
		self.test_path = 'data/test/'

	def pre_train_data(self):
		print('\n\nPrepare Training Data')
		print('-' * 30 + '\nConvert to NPY\n' + '-' * 30)
		count, i= 0, 0
		for indir in os.listdir(self.augmentation_path):
			path = os.path.join(self.augmentation_path, indir)
			count += len(os.listdir(path))
		print('Total     : ' + str(count))
		imgset = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		labelset = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		for indir in os.listdir(self.augmentation_path):
			train = os.path.join(self.train_path, indir + '/')
			label = os.path.join(self.label_path, indir + '/')
			names = os.listdir(train)
			for name in names:
				train_name = name
				label_name = name[0:name.rindex('_')] + '_label.tif'
				img = cv2.imread(train + train_name, cv2.IMREAD_GRAYSCALE)
				lab = cv2.imread(label + label_name, cv2.IMREAD_GRAYSCALE)
				img ,lab = img_to_array(img), img_to_array(lab)
				imgset[i], labelset[i] = img, lab
				i += 1
				print('\rProcessed :', i, end = '')
		np.save('train.npy', imgset)
		np.save('label.npy', labelset)

	def pre_test_data(self):
		print('\nPrepare Test Data')
		print('-' * 30 + '\nConvert to NPY\n' + '-' * 30)
		names = os.listdir(self.test_path)
		imgset = np.ndarray((len(names), self.out_rows, self.out_cols, 1), dtype=np.uint8)
		print('Total   Image :',len(names))
		i = 0
		for name in names:
			img = cv2.imread(self.test_path + name, cv2.IMREAD_GRAYSCALE)
			imgset[i] = img_to_array(img)
			i += 1
			print('\rProcess Image :', i, end = '')
		np.save('test.npy', imgset)

	def load_train(self):
		print('\n\n' + '-' * 30 + '\nLoad Training Data\n' + '-' * 30)
		train = (np.load('train.npy')).astype('float32')
		label = (np.load('label.npy')).astype('float32')
		train = self.norm_train(train)
		label = self.norm_label(label)
		print('OK')
		return self.norm_train(train), self.norm_label(label)

	def load_test(self):
		print('\n\n' + '-' * 30 + '\nLoad Test Data\n' + '-' * 30)
		test_set = (np.load('test.npy')).astype('float32')
		print('OK')
		return self.norm_train(test_set)

	def load_result(self):
		orig = self.load_test()
		orig = orig.reshape(np.shape(orig)[:-1])
		print('\n\n' + '-' * 30 + '\nLoad Test Result\n' + '-' * 30)
		re = (np.load('test_result.npy')).astype('float32')
		re = re.reshape(np.shape(re)[:-1])
		for i in range(np.shape(re)[0]):
			plt.subplot(122)
			plt.imshow(self.norm_label(re[i,:,:]), cmap='gray')
			plt.subplot(121)
			plt.imshow(orig[i,:,:], cmap='gray')
			plt.show()

	def norm_train(self, train):
		train /= np.max(train)
		me = train.mean(axis = 0)
		return train - me

	def norm_label(self, label):
		label /= np.max(label)
		label[label>0.5]=1
		label[label<=0.5]=0
		return label


if __name__ == '__main__':
	p = Preparation()
	q = Process()
#	q.load_result()
	p.merge()
	p.spilt()
#	q.pre_train_data()
#	q.pre_test_data()
