import keras
from keras.models import *
from keras.layers import *
from augment import Process
from keras.callbacks import *
from keras.optimizers import *

class NN(object):
	def __init__(self, row, col):
		self.row = row
		self.col = col

	def load_data(self):
		data = Process(self.row, self.col)
		train, label = data.load_train()
		test = data.load_test()
		return train, label, test

	def net_structure(self, show = True):
		print('\n\n' + '-' * 30 + '\nConstruct Neural Network\n' + '-' * 30)
		# Structure
		# input layer
		inputs = Input((self.row, self.col, 1))

		# the first convolution layer-----filters: 64 3x3  +  64 3x3
		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		# the second convolution layer----filters: 128 3x3  +  128 3x3
		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		# the third convolution layer-----filters: 256 3x3  +  256 3x3
		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		# the forth convolution layer-----filters: 512 3x3  +  512 3x3
		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		# the fifth convolution layer-----filters: 1024 3x3  +  1024 3x3
		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		# the sixth convolution layer-----filters: 512 3x3  +  512 3x3------concatenate with the fourth layer
		up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
		merge6=concatenate([drop4,up6],axis=3)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

		# the seventh convolution layer-----filters: 256 3x3  +  256 3x3------concatenate with the third layer
		up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
		merge7 =concatenate([conv3,up7],axis=3)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

		# the eighth convolution layer-----filters: 128 3x3  +  128 3x3------concatenate with the second layer
		up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
		merge8 =concatenate([conv2,up8],axis=3)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

		# the ninth convolution layer-----filters: 64 3x3  +  64 3x3  + 2 3x3------concatenate with the first layer
		up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
		merge9 =concatenate([conv1,up9],axis=3)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
		conv9 = Conv2D(2 , 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

		# output layer------segmentation result------don't change the size of conv9 output
		conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

		# Close-out
		model = Model(inputs=inputs, outputs=conv10)
		model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
		if show:
			print('\nStructure')
			print(model.summary())
		return model

	def train_test(self):
		train, label, test = self.load_data()
		model = self.net_structure()
		print('\n\nTraining Neural Network')
		model_checkpoint = ModelCheckpoint('net.h5', monitor='loss', verbose=1, save_best_only=True)
		model.fit(train, label, batch_size=20, epochs=30, verbose=1, shuffle=True, callbacks=[model_checkpoint])
		test_result = model.predict(test, batch_size=1, verbose=1)
		np.save('test_result.npy', test_result)

if __name__ == '__main__':
	n = NN()
	n.train_test()
