import numpy as np
import struct
import ipdb         # use ipdb.set_trace() to debug
import cv2
import glob			# for importing all pics in a folder
from cleanData import getDataLabels, getBalancedDataLabels
from termcolor import colored 	# for printing to terminal in colored text
import re 			# for removing text from strings (for sending messages to iphone)
import time


# create a Matlab like struct class
class MATLABstruct():
	pass

# log performance of each type of run while tuning parameters
def logPerformance(params):
	# Format of log:
	# min test error, train error at min test error epoch, epoch min test error acheived, preprocessing (raw=0, prep=1), balanced train, balanced test, res = num_row*num_col, num color channels, color_channel_merged, num rotations, num_net_layers, total epochs run, alpha0, alphaf, date/time,
	global nn_log, file_name

	min_test_error_epoch = np.argmin(nn_log[:,1])
	min_test_error = nn_log[min_test_error_epoch,1]
	img_res = params.dim[0]*params.dim[1]

	time_string = str(time.strftime("%m.%d.%Y-- %H:%M:%S, "))	# get string of date and time and add it to the log 

	log = str(min_test_error) + ',' +  str(nn_log[min_test_error_epoch,2]) + ',' + str(min_test_error_epoch) + ',' + params.preprocess + ',' + str(params.balanced_train_set) + ',' + str(params.balanced_test_set) + ',' + str(img_res) + ',' + str(params.dim[2]) + ',' + str(params.merge_channels) + ','	+  str(params.num_rotations) + ',' + str(len(params.layers)) + ',' + str(params.epochs) + ',' + str(params.alpha[0]) + ',' + str(params.alpha[-1]) + ',' + time_string  

	# # format string and replace some text 
	# message[f_indx] = re.sub('\.png$', '', message[f_indx])		# this line finds the substring '\.png$' and replaces it with ''
	# message[f_indx] = re.sub('database/', '', message[f_indx])
	# message[f_indx] = re.sub('_', ' ', message[f_indx])
	

	print colored(log,'red')

	# write to file. use type 'a' to append or 'w' to overwrite
	with open(file_name, "a") as file:
		file.write((log+'\n'))


# load images for test and training set
def parse_images(path, options):
	channels = options.dim[2]	# number of color channels to use
	dim = options.dim[0:2]		# resolution of image to use
	num_rotations = np.max((options.num_rotations,1))	
	data = np.array([])
	for i, img in enumerate(glob.glob(path)):

		if channels == 1:
			input_img = cv2.imread(img,0)	# load image in gray scale
			input_img = np.expand_dims(input_img,axis=2)
		else:
			input_img = cv2.imread(img)	# load image in RGB
			if channels > 3:	# just a check 
				channels = 3

		# add all 3 color channels of each image to the dataset as an example
		for c in range(channels):
			img_channel = input_img[:,:,c]
			img_channel = cv2.resize(img_channel,dim,interpolation = cv2.INTER_AREA)
			# normalize the pixel data by dividing by 255. you can also subtract the min pixel value from each image individually
			if options.normalize:
				img_channel = (img_channel - np.min(img_channel))/255.0
			# img_channel = img_channel/255.0

			for r in range(num_rotations):
				if options.merge_channels and channels==3:
					# ipdb.set_trace()
					img_resized = cv2.resize(input_img,dim,interpolation = cv2.INTER_AREA)
					if options.normalize:
						img_resized = (img_resized - np.min(img_resized))/255.0
					array2vec = img_resized.reshape(np.prod(options.dim),)

				else:
					img_channel = np.fliplr(img_channel.T)

					row,col = img_channel.shape
					array2vec = img_channel.reshape(np.multiply(row,col),)

				if data.shape[0] == 0:
					data = array2vec
				else:
					data = np.vstack((data, array2vec))	# reshape image i into a vector and load into row i in data matrix

			if options.merge_channels:
				# c = range(channels)[-1]
				break

	return data

def parse_labels(options):
	channels = 1 if options.merge_channels else options.dim[2]

	names, labels = getDataLabels()

	if options.balanced_train_set:
		y_tr = getBalancedDataLabels('train')
	else:	
		y_tr = labels.train

	if options.balanced_test_set:
		y_te = getBalancedDataLabels('test')
	else:
		y_te = labels.test

	# repeat vector labels to match the number of color channels used in the RGB image
	y_tr = np.repeat(y_tr,channels*options.num_rotations)
	y_te = np.repeat(y_te,channels*options.num_rotations)

	# format labels to have 2 classes: [1 0] = benign and [0 1] = malignant
	y_train = np.hstack((y_tr.reshape(y_tr.shape[0],1),np.zeros((y_tr.shape[0],1))))	# make y a k-dimensional array where k = # classes = 2 in this problem
	y_train[y_tr == 0] = (1,0)	
	y_train[y_tr == 1] = (0,1)

	y_test = np.hstack((y_te.reshape(y_te.shape[0],1),np.zeros((y_te.shape[0],1))))		# make y a k-dimensional array where k = # classes = 2 in this problem
	y_test[y_te == 0] = (1,0)
	y_test[y_te == 1] = (0,1)
	return y_train, y_test



def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != np.argmax(y,axis=1)))/y.shape[0]


# HELPER FUNCTIONS for loss and neural network activations
softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                              np.exp(yp)/np.sum(np.exp(yp)) - y)
f_tanh = lambda x : (np.tanh(x), 1./np.cosh(x)**2)
f_relu = lambda x : (np.maximum(0,x), (x>=0).astype(np.float64))
f_lin = lambda x : (x, np.ones(x.shape))


# other loss functions
l2_norm = lambda x : np.sum(np.abs(x)**2,axis=-1)**(1./2)
# indicator  = lambda y, yp : np.asarray([1 if (y[x,1] != yp[x,1] and y[x,1] == 1) else 0 for x in range(y.shape[0])])
indicator  = lambda yp, y : np.array([1 if (y[1] != yp[1] and y[1] == 1) else 0])


##### Implement the functions below this point ######

def softmax_gd(X, y, Xt, yt, epochs=10, alpha = 0.5):
	""" 
	Run gradient descent to solve linear softmax regression.

	Inputs:
		X: numpy array of training inputs
		y: numpy array of training outputs
		Xt: numpy array of testing inputs
		yt: numpy array of testing outputs
		epochs: number of passes to make over the whole training set
		alpha: step size
	    
	Outputs:
	    Theta: 10 x 785 numpy array of trained weights
	"""

	# Theta is a 10 x 785 vector of parameters. initialize theta as a 10 x 785 array of zeros
	n = np.shape(y)[1]
	m = np.shape(X)[1]+1
	train_num = np.shape(X)[0]
	test_num = np.shape(Xt)[0]
	theta = np.zeros((n,m))
	y_hat = np.zeros(np.shape(y))
	y_hat_t = np.zeros(np.shape(yt))
	loss = np.zeros(train_num)
	loss_t = np.zeros(test_num)
	err = np.zeros(epochs)

	# append a column of ones to training data set X. 
	x_i = np.hstack(( X, np.ones((train_num,1)) ))

	# append a column of ones to testing data set Xt. 
	xt_i = np.hstack(( Xt, np.ones((test_num,1)) ))

	print 'test err  | train err  | test loss   | train loss   |'

	# loop through the whole training set a few (epoch) times
	for t in range(epochs):
		# initialize the gradient vector (same size as theta)
		g = np.zeros((n,m))

		for i in range(train_num):
				
			# TRAINING DATA
			# calculate the hypothesis function values
			y_hat[i,:] = np.dot(theta, x_i[i,:])		# this is y_hat
			loss[i] = softmax_loss(y_hat[i,:], y[i,:])[0] 
			loss_gradient = softmax_loss(y_hat[i,:], y[i,:])[1]
			g = g + (1/float(train_num))*(np.outer(loss_gradient, np.transpose(x_i[i,:]) ))

			if i < test_num:
				# TESTING DATA
				y_hat_t[i,:] = np.dot(theta, xt_i[i,:])		# this is y_hat
				loss_t[i] = softmax_loss(y_hat_t[i,:], yt[i,:])[0] 


		# report training errors
		train_loss_avg = np.mean(loss)

		train_err = error(y_hat, y)

		# report testing errors
		test_loss_avg = np.mean(loss_t)

		test_err = error(y_hat_t, yt)

		# update theta
		theta = theta - alpha*g

		print test_err, ',', train_err, ',', test_loss_avg, ',', train_loss_avg

	return theta



def softmax_sgd(X,y, Xt, yt, epochs=10, alpha = 0.01):
	""" 
	Run stoachstic gradient descent to solve linear softmax regression.

	Inputs:
		X: numpy array of training inputs
		y: numpy array of training outputs
		Xt: numpy array of testing inputs
		yt: numpy array of testing outputs
		epochs: number of passes to make over the whole training set
		alpha: step size
	    
	Outputs:
		Theta: 2 x 785 numpy array of trained weights
    """

	# Theta is a 2 x 785 vector of parameters. initialize theta as a 10 x 785 array of zeros
	n = np.shape(y)[1]			# number of classes
	m = np.shape(X)[1]+1		# number of features per example
	train_num = np.shape(X)[0]	# number of training examples
	test_num = np.shape(Xt)[0]	# number of test examples

	theta = np.zeros((n,m))
	y_hat = np.zeros(np.shape(y))
	y_hat_t = np.zeros(np.shape(yt))
	loss = np.zeros(train_num)
	loss_t = np.zeros(test_num)
	err = np.zeros(epochs)

	# append a column of ones to training data set X. 
	x_i = np.hstack(( X, np.ones((train_num,1)) ))

	# append a column of ones to testing data set Xt. 
	xt_i = np.hstack(( Xt, np.ones((test_num,1)) ))

	print 'test err  | train err  | test loss   | train loss   |'

	# loop through the whole training set a few (epoch) times
	for t in range(epochs):
		# initialize the gradient vector (same size as theta)
		g = np.zeros((n,m))

		for i in range(train_num):
				
			# TRAINING DATA
			# calculate the hypothesis function values
			y_hat[i,:] = np.dot(theta, x_i[i,:])		# this is y_hat
			loss[i] = softmax_loss(y_hat[i,:], y[i,:])[0] 
			loss_gradient = softmax_loss(y_hat[i,:], y[i,:])[1]
			theta = theta - alpha*(np.outer(loss_gradient, np.transpose(x_i[i,:]) ))

			if i < test_num:
				# TESTING DATA
				y_hat_t[i,:] = np.dot(theta, xt_i[i,:])		
				loss_t[i] = softmax_loss(y_hat_t[i,:], yt[i,:])[0] 


		# report training errors
		train_loss_avg = np.mean(loss)

		train_err = error(y_hat, y)

		# report testing errors
		test_loss_avg = np.mean(loss_t)

		test_err = error(y_hat_t, yt)

		# update theta
		theta = theta - alpha*g

		print test_err, ',', train_err, ',', test_loss_avg, ',', train_loss_avg

	return theta


def nn(x, W, b, f):
	"""
	Compute output of a neural network.

	Input:
		x: numpy array of input
		W: list of numpy arrays for W parameters
		b: list of numpy arrays for b parameters
		f: list of activation functions for each layer
	    
	Output:
		z: list of activations, where each element in the list is a tuple:
			(z_i, z'_i)
			for z_i and z'_i each being a numpy array of activations/derivatives
	"""
	# generate zi and zi_prime: the list of features and derivative of the features
	z = list()
	zp = list()			# zp is the derivative of the feature set or activations
	k = len(W) + 1  	# W is applied during every transition, therefore the number of layers (k) is one more 

	# initialize z1 to be equal to x
	z.append(x)
	zp.append(x)

	# generate z_(i+1), z'_(i+1)
	for i in range(k-1):
		# grab function from function list
		fi = f[i]
		zi = np.reshape(z[i], (len(z[i]),1))
		bi = np.reshape(b[i], (len(b[i]),1))

		# use the function to compute z_(i+1), z'_(i+1). Then append to the z and zp lists
		z_ip1, zp_ip1 = fi(np.dot(W[i],zi) + bi)
		z.append(z_ip1)
		zp.append(zp_ip1)

	return z, zp


def nn_loss(x, y, W, b, f):
	"""
	Compute loss of a neural net prediction, plus gradients of parameters

	Input:
		x: numpy array of input
		y: numpy array of output
		W: list of numpy arrays for W parameters
		b: list of numpy arrays for b parameters
		f: list of activation functions for each layer
	    
	Output tuple: (L, dW, db)
		L: softmax loss on this example
		dW: list of numpy arrays for gradients of W parameters
		db: list of numpy arrays for gradients of b parameters
	"""

	# use back propogation to calculate the gradients of W and b and the loss
	k = len(W) + 1  	# W is applied during every transition, therefore the number of layers (k) is one more 
	g = [None]*k
	db = [None]*k
	dW = [None]*k

	# call nn to get z and z prime
	z, zp = nn(x, W, b, f)

	# compute the loss (L) and the gradient of the loss (g_k)
	yp = z[-1].flatten()
	L, g_k = softmax_loss(yp, y)
	# L += indicator(yp,y)
	# g_k += indicator(yp,y)*[.1,-.1]


	# g_k is the last (k'th) gradient in the list of gradients so insert it into g
	g[-1] = g_k

	
	for i in reversed(range(k-1)):
		g[i] = np.dot(W[i].T, np.multiply(g[i+1],zp[i+1].flatten())) 
		db[i] = np.multiply(g[i+1],zp[i+1].flatten())
		dW[i] = np.outer(np.multiply(g[i+1],zp[i+1].flatten()), z[i].T)

	return L, db, dW, z


def nn_sgd(X,y, Xt, yt, W, b, f, epochs=10, alpha = 0.01):
	""" 
	Run stoachstic gradient descent to solve linear softmax regression.

	Inputs:
		X: numpy array of training inputs
		y: numpy array of training outputs
		Xt: numpy array of testing inputs
		yt: numpy array of testing outputs
		W: list of W parameters (with initial values)
		b: list of b parameters (with initial values)
		f: list of activation functions
		epochs: number of passes to make over the whole training set
		alpha: step size

	Output: None (you can directly update the W and b inputs in place)
	"""
	global nn_log, params

	# Theta is a 10 x 785 vector of parameters. initialize theta as a 10 x 785 array of zeros
	k = len(W) + 1
	train_num = np.shape(X)[0]
	test_num = np.shape(Xt)[0]
	L = np.zeros(train_num)
	L_t = np.zeros(test_num)
	z = np.zeros(y.shape)
	z_t = np.zeros(yt.shape)
	err = np.zeros(epochs)

	print 'test err  | train err  | test loss   | train loss   |'
	
	# dynamically change alpha for speed
	if alpha.shape[0] < epochs:
		alpha = np.repeat(alpha,epochs/alpha.shape[0])


	for t in range(epochs):
		alpha_d = alpha[t]

		for i in range(train_num):

			L[i], db, dW, zz = nn_loss(X[i], y[i], W, b, f)
			z[i,:] = zz[-1].T
			for j in range(k-1):	# iterate through number of neural network layers (k)

				# regularization term
				regTerm = params.regularize * np.linalg.norm(np.asarray(W[j]))

				W[j] = W[j] - alpha_d*(dW[j] + regTerm)
				b[j] = b[j] - alpha_d*db[j]

			if i < test_num:
				L_t[i], db_t, dW_t, zz_t = nn_loss(Xt[i], yt[i], W, b, f)
				z_t[i,:] = zz_t[-1].T

		# report training errors
		train_loss_avg = np.mean(L)

		train_err = error(z, y)

		# report testing errors
		test_loss_avg = np.mean(L_t)

		test_err = error(z_t, yt)

		print test_err, ',', train_err, ',', test_loss_avg, ',', train_loss_avg

		# log performance data
		if nn_log.shape[0]==0:
			nn_log = np.array([t, test_err, train_err, test_loss_avg, train_loss_avg])
		else:
			nn_log = np.vstack((nn_log,[t, test_err, train_err, test_loss_avg, train_loss_avg]))

	return W, b






def main():
	# define global variables
	global params, file_name, nn_log
	file_name = 'logs/performance_log.csv'
	nn_log = np.array([])

	params = MATLABstruct()
	options = MATLABstruct()
	params.dim = options.dim = (20,20,3)		# image dimension (resolution_row, resolution_col, number of color channels)
	params.merge_channels = options.merge_channels = True	# False: uses number of channels as a separate training example, True: multiplies each training example's feature size by number of channels
	params.num_rotations = options.num_rotations = 2	# number of image rotations (increase data set by rotating each image 90 deg and counting it as a new training example)
	params.preprocess = 'processed'						# 'processed' for processed set and 'unprocessed' for unprocessed set
	params.balanced_train_set = options.balanced_train_set = False		# True for balanced set and False for original unbalanced set
	params.balanced_test_set = options.balanced_test_set = True
	options.normalize = True
	params.regularize = 0.000000
	params.epochs = 30
	alpha0 = .05; alphaf = .001
	params.alpha = np.linspace(alpha0,alphaf,params.epochs)

	add2path_balanced_tr = '/balanced' if options.balanced_train_set else ''
	add2path_balanced_te = '/balanced' if options.balanced_test_set else ''

	# load data from file
	# X_train = num_training_images x dim numpy array where each row corresponds to a 28 by 28 image of a digit
	# y_train = num_training_images x 2 array where each row is an indicator of which digit is present in the image
	X_train = parse_images('data/formatted/train/' + params.preprocess + add2path_balanced_tr + '/*.jpeg',options)
	y_train = parse_labels(options)[0]	# get training labels

	options.num_rotations = 1
	X_test = parse_images('data/formatted/test/' + params.preprocess + add2path_balanced_te + '/*.jpeg',options)
	y_test = parse_labels(options)[1]	# get test labels


	# Set up a simple deep neural network for melanoma classification
	# Initialize with random weight, with rectified linear units in all layers
	# except the last layer which uses a linear unit
	#
	# Create a deep network with 4 layers (2 hidden layers). 
	# Layer 1 size: 65536 (256 x 256: size of input)
	# Layer 2 size: 1000
	# Layer 3 size: 100
	# Layer 4 size: 2 (size of output)
	#
	# A network with i layers will have i-1 W terms or transitions
	# W_1 is 200 x 784
	# W_2 is 100 x 200
	# W_3 is 10 x 100
	np.random.seed(0)
	params.layers = [X_train.shape[1], X_train.shape[1]//2, X_train.shape[1]//4, X_train.shape[1]//8, 2]
	# params.layers = [X_train.shape[1], X_train.shape[1]//4, X_train.shape[1]//8, 2]
	W = [0.1*np.random.randn(n,m) for m,n in zip(params.layers[:-1], params.layers[1:])]
	b = [0.1*np.random.randn(n) for n in params.layers[1:]]
	f = [f_relu]*(len(params.layers)-2) + [f_lin]
	# trial example: [784, 200, 100, 10, 2] alpha = dynamic .1:.01:.01

	# # plot image
	# M,N = 10,20
	# fig, ax = plt.subplots(figsize=(N,M))
	# digits = np.vstack([np.hstack([np.reshape(X_train[i*N+j,:],(28,28)) for j in range(N)]) for i in range(M)])
	# ax.imshow(255-digits, cmap=plt.get_cmap('gray'))



	# # create sub samples (ss) for quicker debugging for the neural network code
	# ss_size = 100
	# X_train_ss = X_train[0:ss_size,:]
	# y_train_ss = y_train[0:ss_size,:]
	# X_test_ss = X_test[0:ss_size,:]
	# y_test_ss = y_test[0:ss_size,:]

	# RUN DIFFERENT ALGORITHMS HERE:
	# theta = softmax_gd(X_train, y_train, X_test, y_test)
	# theta = softmax_sgd(X_train, y_train, X_test, y_test,30)

	# nn(X_train_ss[0], W, b, f)
	# nn_loss(X_train_ss[0], y_train_ss[0], W, b, f)
	# nn_sgd(X_train_ss[0:1000], y_train_ss[0:1000],X_test_ss[0:1000], y_test_ss[0:1000], W, b, f)
	nn_sgd(X_train, y_train,X_test, y_test, W, b, f, params.epochs, params.alpha)

	# create parameters struct
	params.nn_log = nn_log
	logPerformance(params)



if __name__ == "__main__":
	main()


