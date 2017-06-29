'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from cleanData import getDataLabels, getBalancedDataLabels
import numpy as np
import glob
import cv2
import ipdb

class MATLABstruct():
    pass


# load images for test and training set
def parse_images(path, options):
    # define default params if not initialized
    options.normalize = options.normalize if hasattr(options, 'normalize') else True
    options.data_3d = options.data_3d if hasattr(options, 'data_3d') else False

    channels = options.dim[2]   # number of color channels to use
    dim = options.dim[0:2]      # resolution of image to use 
    data = [] if hasattr(options, 'data_3d') else np.array([]) 

    for i, img in enumerate(glob.glob(path)):

        if options.data_3d:
            input_img = cv2.imread(img) # load image in RGB
            img_resized = cv2.resize(input_img,dim,interpolation = cv2.INTER_AREA)
            if options.normalize:
                img_resized = (img_resized - np.min(img_resized))/255.0
            data.append(img_resized)

    data = np.asarray(data) if hasattr(options, 'data_3d') else data 
    return data


options = MATLABstruct()
options.dim = (50,50,3)        # image dimension (resolution_row, resolution_col, number of color channels)
options.preprocess = 'processed'                     # 'processed' for processed set and 'unprocessed' for unprocessed set
options.balanced_train_set = True      # True for balanced set and False for original unbalanced set
options.balanced_test_set = True
options.data_3d = True           # True: parse_images function will output a data set with 3d examples (for CNN) False: parse_images function outputs a 1-D example for NN
options.normalize = False     # normalizing subtracts min pixel value from all pixels in image and divides by 255

add2path_balanced_tr = '/balanced' if options.balanced_train_set else ''
add2path_balanced_te = '/balanced' if options.balanced_test_set else ''


batch_size = 32
nb_classes = 2
nb_epoch = 50
data_augmentation = True

# input image dimensions
img_rows, img_cols = options.dim[0:2]
img_channels = 3    # the melanoma images are RGB.

# Load data from cifar10: The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# load data from file
# X_train = num_training_images x dim numpy array where each row corresponds to a 28 by 28 image of a digit
# y_train = num_training_images x 2 array where each row is an indicator of which digit is present in the image
X_train = parse_images('data/formatted/train/' + options.preprocess + add2path_balanced_tr + '/*.jpeg',options)
X_test = parse_images('data/formatted/test/' + options.preprocess + add2path_balanced_te + '/*.jpeg',options)
names, labels = getDataLabels()
y_train = getBalancedDataLabels('train') if options.balanced_train_set else labels.train  # get training labels
y_test = getBalancedDataLabels('test') if options.balanced_test_set else labels.test   # get test labels

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# ipdb.set_trace()

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))


