from keras import backend as K

K.set_image_dim_ordering('th')

# build model in TH mode, as th_model
th_model = ...
# load weights that were saved in TH mode into th_model
th_model.load_weights(...)

K.set_image_dim_ordering('tf')

# build model in TF mode, as tf_model
tf_model = ...

# transfer weights from th_model to tf_model
for th_layer, tf_layer in zip(th_model.layers, tf_model.layers):
   if th_layer.__class__.__name__ == 'Convolution2D':
      kernel, bias = layer.get_weights()
      kernel = np.transpose(kernel, (2, 3, 1, 0))
      tf_layer.set_weights([kernel, bias])
  else:
      tf_layer.set_weights(tf_layer.get_weights())