import tensorflow as tf
import numpy as np

# Deep neural network: 4 hidden layers fully connected with relu activation function and output layer with one unit and sigmoid activation

def DNN(input_shape):
    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Flatten()(X_input)
    X = tf.keras.layers.Dense(512, activation='relu', name='fc0', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(X)
    X = tf.keras.layers.Dense(256, activation='relu', name='fc1', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(X)
    X = tf.keras.layers.Dense(128, activation='relu', name='fc2', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(X)
    X = tf.keras.layers.Dense(64, activation='relu', name='fc3', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc4')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


# Convolutional network with 3 convolutional layers, batch normalizations and dropouts

def CNN(input_shape):
    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=2)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(X)
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


# Convolutional network with 3 convolutional layers dilated, batch normalizations and dropouts

def CNNd(input_shape):
    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name='conv0')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=2)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


# Locally connected network with three hidden locally connected layers, batch normalizations and dropouts

def CNNl(input_shape):
    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.LocallyConnected2D(filters=64, kernel_size=(3, 3))(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.LocallyConnected2D(filters=128, kernel_size=(3, 3))(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.LocallyConnected2D(filters=256, kernel_size=(3, 3))(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=2)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


# Recurrent neural network with GRU after a convolutional network

def RNN(input_shape):
    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=1, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(X)
    X = tf.keras.layers.Reshape(target_shape=((X.shape[1] * X.shape[2]), X.shape[3]))(X)

    # GRU
    X = tf.keras.layers.GRU(units=1024)(X)

    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


# Convolutional autoencoder

def DAE(input_shape, code_size):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(input_shape))

    encoder.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same', name='conv0'))
    encoder.add(tf.keras.layers.MaxPool2D((1, 2), padding='same'))
    encoder.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', name='conv1'))
    encoder.add(tf.keras.layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(code_size))
    encoder.build()

    # decoder
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.Input((code_size,)))
    if (input_shape[0] == 32):
        shape = (8, np.int(np.ceil(np.prod(input_shape) / (256 * 2))), 256)
    else:
        shape = (15, np.int(np.prod(input_shape) / (240 * 4)), 256)
    decoder.add(tf.keras.layers.Dense(np.prod(shape), activation='elu'))
    decoder.add(tf.keras.layers.Reshape(shape))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same', name='convT0'))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=2, activation=None, padding='same', name='convT1'))
    decoder.build()

    return encoder, decoder


# Convolutional autoencoder for FFT (a different reshape and stride is needed for the dataset dimension)

def DAE_FFT(input_shape, code_size):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(input_shape))

    encoder.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='elu', padding='same', name='conv0'))
    encoder.add(tf.keras.layers.MaxPool2D((1, 2), padding='same'))
    encoder.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='elu', padding='same', name='conv1'))
    encoder.add(tf.keras.layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(code_size))
    encoder.build()

    # decoder
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.Input((code_size,)))
    shape = (32, 513, 256)
    decoder.add(tf.keras.layers.Dense(np.prod(shape), activation='elu'))
    decoder.add(tf.keras.layers.Reshape(shape))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(1, 1), strides=1, activation='elu', padding='same', name='convT2'))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(1, 1), strides=1, activation='elu', padding='same', name='convT0'))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(1, 1), strides=1, activation=None, padding='same', name='convT1'))
    decoder.build()

    return encoder, decoder


# Deep neural network build after each model, removing the last layer and inserting four fully-connected layers

def DNN_model(autoencoder):
    model = tf.keras.models.Sequential()
    for layer in autoencoder.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(tf.keras.layers.Dense(units=256, activation='relu', name='fc0', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', name='fc1', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
    model.add(tf.keras.layers.Dense(units=64, activation='relu', name='fc2', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', name='fc3'))

    model.build()
    return model




