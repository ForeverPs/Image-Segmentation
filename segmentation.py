from net import *
from augment import *
from con2tif import *

if __name__ == '__main__':
	C = Convert()
	C.nii2tif()
	P = Preparation()
	Q = Process(880,880)
	P.merge()
	P.spilt()
	Q.pre_train_data()
	Q.pre_test_data()
	N = NN(880,880)
	N.train_test()
	Q.load_result()
