import numpy as np
import struct
import ipdb         # use ipdb.set_trace() to debug
import matplotlib.pyplot as plt


def parse_images(filename):
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    sx,sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im =  struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/255.0 for x in im]);
    return np.array(X);

def parse_labels(filename):
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=np.float64)
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    return one_hot(np.array(struct.unpack('B'*size, f.read(size))), 10)

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != np.argmax(y,axis=1)))/y.shape[0]


# load data from file
# X_train = 60,000 x 784 numpy array where each row corresponds to a 28 by 28 image of a digit
# y_train = 60,000 x 10 array where each row is an indicator of which digit is present in the image
X_train = parse_images("train-images-idx3-ubyte")
y_train = parse_labels("train-labels-idx1-ubyte")
X_test = parse_images("t10k-images-idx3-ubyte")
y_test = parse_labels("t10k-labels-idx1-ubyte")


# helper functions for loss and neural network activations
softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                              np.exp(yp)/np.sum(np.exp(yp)) - y)
f_tanh = lambda x : (np.tanh(x), 1./np.cosh(x)**2)
f_relu = lambda x : (np.maximum(0,x), (x>=0).astype(np.float64))
f_lin = lambda x : (x, np.ones(x.shape))


# Set up a simple deep neural network for MNIST task. 
# Initialize with random weight, with rectified linear units in all layers
# except the last layer which uses a linear unit
#
# Create a deep network with 4 layers (2 hidden layers). 
# Layer 1 size: 784 (size of input)
# Layer 2 size: 200
# Layer 3 size: 100
# Layer 4 size: 10 (size of output)
#
# A network with i layers will have i-1 W terms or transitions
# W_1 is 200 x 784
# W_2 is 100 x 200
# W_3 is 10 x 100
np.random.seed(0)
layer_sizes = [784, 200, 100, 10]
W = [0.1*np.random.randn(n,m) for m,n in zip(layer_sizes[:-1], layer_sizes[1:])]
b = [0.1*np.random.randn(n) for n in layer_sizes[1:]]
f = [f_relu]*(len(layer_sizes)-2) + [f_lin]

# t1: [784, 200, 100, 10]
# t2: [784, 500, 250, 100, 50, 10]
# t3: [784, 200, 10]
# t4: [784, 200, 100, 10] alpha = .1
# t5: [784, 200, 100, 10] f = [f_lin f_lin f_lin f_lin] linear for all layers
# t6: [784, 200, 100, 10] alpha = dynamic .1:.01:.01



# # plot image
# M,N = 10,20
# fig, ax = plt.subplots(figsize=(N,M))
# digits = np.vstack([np.hstack([np.reshape(X_train[i*N+j,:],(28,28)) for j in range(N)]) for i in range(M)])
# ax.imshow(255-digits, cmap=plt.get_cmap('gray'))

ipdb.set_trace()



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
			y_hat[i][:] = np.dot(theta, x_i[i][:])		# this is y_hat
			loss[i] = softmax_loss(y_hat[i][:], y[i][:])[0] 
			loss_gradient = softmax_loss(y_hat[i][:], y[i][:])[1]
			g = g + (1/float(train_num))*(np.outer(loss_gradient, np.transpose(x_i[i][:]) ))

			if i < test_num:
				# TESTING DATA
				y_hat_t[i][:] = np.dot(theta, xt_i[i][:])		# this is y_hat
				loss_t[i] = softmax_loss(y_hat_t[i][:], yt[i][:])[0] 


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
			y_hat[i][:] = np.dot(theta, x_i[i][:])		# this is y_hat
			loss[i] = softmax_loss(y_hat[i][:], y[i][:])[0] 
			loss_gradient = softmax_loss(y_hat[i][:], y[i][:])[1]
			theta = theta - alpha*(np.outer(loss_gradient, np.transpose(x_i[i][:]) ))

			if i < test_num:
				# TESTING DATA
				y_hat_t[i][:] = np.dot(theta, xt_i[i][:])		
				loss_t[i] = softmax_loss(y_hat_t[i][:], yt[i][:])[0] 


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

# # run the function to test GD or SGD functions
# theta = softmax_gd(X_train, y_train, X_test, y_test)
# theta = softmax_sgd(X_train, y_train, X_test, y_test)



# # create sub samples (ss) for quicker debugging for the neural network code
ss_size = 1000
X_train_ss = X_train[0:ss_size][:]
y_train_ss = y_train[0:ss_size][:]
X_test_ss = X_test[0:ss_size][:]
y_test_ss = y_test[0:ss_size][:]


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
	L, g_k = softmax_loss(z[-1].flatten(), y)

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

	# Theta is a 10 x 785 vector of parameters. initialize theta as a 10 x 785 array of zeros
	k = len(W) + 1
	train_num = np.shape(X)[0]
	test_num = np.shape(Xt)[0]
	L = np.zeros(train_num)
	L_t = np.zeros(test_num)
	z = np.zeros((train_num,10))
	z_t = np.zeros((test_num,10))
	err = np.zeros(epochs)

	print 'test err  | train err  | test loss   | train loss   |'
	
	# dynamically change alpha for speed
	# alpha_d = np.linspace(.1,.01,10)   

	for t in range(epochs):
		# alpha = alpha_d[t]

		for i in range(train_num):
			# ipdb.set_trace()
			L[i], db, dW, zz = nn_loss(X[i], y[i], W, b, f)
			z[i][:] = zz[-1].T
			for j in range(k-1):
				W[j] = W[j] - alpha*dW[j]
				b[j] = b[j] - alpha*db[j]


			if i < test_num:
				L_t[i], db_t, dW_t, zz_t = nn_loss(Xt[i], yt[i], W, b, f)
				z_t[i][:] = zz_t[-1].T



		# report training errors
		train_loss_avg = np.mean(L)

		train_err = error(z, y)

		# report testing errors
		test_loss_avg = np.mean(L_t)

		test_err = error(z_t, yt)

		print test_err, ',', train_err, ',', test_loss_avg, ',', train_loss_avg

		# f = open('results.csv','w')
		# f.write(str((test_err, train_err, test_loss_avg, train_loss_avg + '\n'))) # python will convert \n to os.linesep
		# f.close() # you can omit in most cases as the destructor will call it

	return W, b



# nn(X_train_ss[0], W, b, f)
# nn_loss(X_train_ss[0], y_train_ss[0], W, b, f)
# nn_sgd(X_train_ss[0:1000], y_train_ss[0:1000],X_test_ss[0:1000], y_test_ss[0:1000], W, b, f)
nn_sgd(X_train, y_train,X_test, y_test, W, b, f)


