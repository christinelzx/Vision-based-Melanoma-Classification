    # First define the image model
    image_processor = Sequential()
    image_processor.add(Convolution2D(nb_filters, filter_row,     filter_col, image_shape=image_shape))
    image_processor.add(Activation('relu'))
    ... # you may want more layers here, like max_pooling and such
    image_processor.add(Flatten())  # transform image to vector
     
    # Now we create the metadata model
    metadata_processor = Sequential()
    metadata_processor.add(Dense(output_dim, input_dim=metadata_dim))
    metadata_processor.add(Activation('relu'))
    ... # maybe more?
     
    # Now we concatenate the two features and add a few more layers on top
    model = Sequential()
    model.add(Merge([image_processor, metadata_processor], merge_mode='concat')  # Merge is your sensor fusion buddy
    model.add(Dense(this_dim, input_dim=image_plus_metadata_dim))
    model.add(Activation('relu'))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))