'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
 
import os
import h5py
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from ipdb import set_trace as st
import time

# dimensions of our images.
img_width, img_height = 250, 250

# Generally, the smaller the batch size the noisier the updates,
# If you decrease the batch size you should probably decrease the learning rate, and train for more epochs.

# Parameters and paths to the model weights file.
train_dataset = 'unbalanced'      # 'balanced' or 'unbalanced' as input
test_dataset = 'balanced' 
train_data_dir = 'data/cnn_data/' + train_dataset + '/train/processed'
validation_data_dir = 'data/cnn_data/' + test_dataset + '/test/processed'       
nb_train_samples = 346 if train_dataset == 'balanced' else 900             # (unbalanced = 346) (balanced = 900)
nb_validation_samples = 150 if test_dataset == 'balanced' else 379      # (balanced = 150) (unbalanced = 379)
nb_epoch = 20           # default: 50
batch_size = 32         # default: 32  
learning_rate = 1e-4    # default: 1e-4
weights_path = 'weights/vgg16_weights.h5'
top_model_weights_path = 'weights/top_layer_model_' + str(img_width) + train_dataset + '_' + test_dataset + '-' + time.strftime('%d_%m_%y') + '.h5'
fine_tuned_melanoma_weights = 'weights/melanoma_VGG16-FT_' + str(img_width) + train_dataset + '_' + test_dataset + '-' + time.strftime('%d_%m_%y') + '.h5'

K.set_image_dim_ordering('th')      # set model to be ordered in dimensions for theano

def buildModel_VGG16():
    # build the VGG16 network
    # zero padding: adds a border of thickness (1,1) pixels of zeros around the image. 
    # activation funciton: rectified linear unit does not suffer from gradient vanishing
    # Convolution2D(<number of filters>,<filter_size_dim1>,<filter_size_dim2>,activation = <>, name = <>)
    # Convolutional layers using a stride of 1, filters of size FxF, and zero padding with (F-1)/2 will preserve size spatially
    
    # Convolutional Layer: 
    #   - Input size: W1 x H1 X D1
    #   - Requires 4 hyper parameters: 
    #       * number of filters (K), [common settings are powers of 2: 32, 64, 128, 256, 512]
    #       * their spatial extent/filter size (F), [common settings: F = 3, S = 1, P = 1]
    #       * the stride, or number of pixels the filter is translated (S)
    #       * the amount of zero-padding (P)
    #   - Produces a volume of size W2 X H2 X D2, where:
    #       * W2 = (1/S)*(W1 - F + 2P) + 1
    #       * H2 = (1/S)*(H1 - F + 2P) + 1 (width and height computed equally by symmetry)
    #       * D2 = K
    #   - With parameter sharing, it introduces F*F*D1 weights per filter, for a total of (F*F*D1)*K wieghts and K biases
    #   - In the output volume, the d-th depth slice (of W2 X H2) is the result of performing a valid convolution of the d-th layer over the input
    #     volume with a stride (S) and then offset by the d-th bias
    #
    # Max-pooling Layer:
    #   - Input size: W1 X H1 X D1
    #   - Requires 2 hyper parameters
    #       * their spatial extent/filter size (F)
    #       * the stride, or number of pixels the filter is translated (S)
    #   - Produces a volume of size W2 x H2 X D2, where:
    #       * W2 = (1/S)*(W1 - F) + 1
    #       * H2 = (1/S)*(H1 - F) + 1
    #       * D2 = D1
    #   - Introduces zero parameters since it computes a fixed function of the input
    #   - Not common to use zero-padding for Pooling layers



    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))    
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    return model

def buildModel_TopLayer():
    # load the train and test data and labels from temp
    train_data = np.load(open('temp/bottleneck_features_train.npy'))

    # build a classifier model to put on top of the convolutional model.
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))       # for multi-class classification change  to Dense(num_classes, activation='softmax')
    return model

def buildModel_VGG16FT():
    # fine tune the complete VGG16 model with added top layer for melanoma classification
    model = buildModel_VGG16()
    top_model = buildModel_TopLayer()

    # add the model on top of the convolutional base
    model.add(top_model)

    # load complete set of weights for the fine-tuned network
    model.load_weights(fine_tuned_melanoma_weights)
    return model


def save_bottlebeck_features():
    # Bottleneck features. Use VGG-16 as is to classify data
    global batch_size
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = buildModel_VGG16()

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('temp/bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('temp/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    # load the train and test data and labels from temp
    train_data = np.load(open('temp/bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('temp/bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    # train the top layers of the model which we will then add to the pretrained vgg16 network
    model = buildModel_TopLayer()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

def fine_tune_VGG16():
    # fine tune the complete VGG16 model with added top layer for melanoma classification
    model = buildModel_VGG16()
    top_model = buildModel_TopLayer()

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

    # fine-tune the model
    model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)

    # save the fine-tuned model weights
    model.save_weights(fine_tuned_melanoma_weights)


def classify_image(img_path):
    img = cv2.resize(cv2.imread(img_path), (img_width, img_height))     # load image and resize
    mean_pixel = [103.939, 116.779, 123.68]     # mean pixel values over ImageNet dataset taken from VGG-16 paper 
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]     # normalize the data by subtracting the mean pixel values
    img = img.transpose((2,0,1))    # rearrange image channels for correctly formatted input into the Keras model
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    model = buildModel_VGG16FT()
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    out = model.predict(img)


def main():
    retrain = True
    classify = False
    img_path = ''

    if retrain:
        save_bottlebeck_features()  # generate VGG16 model and upload trianed weights (everything except top layer) 
        train_top_model()           # add top layer to the VGG16 and pre-train the top layer on the new data
        fine_tune_VGG16()           # load pre-trained top layer weights and then fine-tune the network by tuning just the top layer 

    if classify:
        classify_image(img_path)



if __name__ == '__main__':
    main()