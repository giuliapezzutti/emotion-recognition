import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model


def identity_block(X_input, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '1th')(X_input)
    X = BatchNormalization(axis=3, name=bn_name_base + '1th')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '2nd')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2nd')(X)
    X = Activation('relu')(X)

    # Addition of the two
    X = Add()([X_input, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X_input, f, filters, stage, block, s=1):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters

    X = Conv2D(F1, (f, f), strides=(s, s), padding='same', name=conv_name_base + '1st')(X_input)
    X = BatchNormalization(axis=3, name=bn_name_base + '1st')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), strides=(s, s), padding='same', name=conv_name_base + '2nd')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2nd')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '1')(X_input)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def ResNet(input_shape):

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64], stage=2, block='a')
    X = identity_block(X, f=3, filters=[64, 64], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='a')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256], stage=4, block='a')
    X = identity_block(X, 3, [256, 256], stage=4, block='b')
    X = identity_block(X, 3, [256, 256], stage=4, block='c')
    X = identity_block(X, 3, [256, 256], stage=4, block='d')
    X = identity_block(X, 3, [256, 256], stage=4, block='e')
    X = identity_block(X, 3, [256, 256], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512], stage=5, block='a')
    X = identity_block(X, 3, [512, 512], stage=5, block='b')
    X = identity_block(X, 3, [512, 512], stage=5, block='c')

    if X.shape[2]!=1:
        X = tf.keras.layers.AveragePooling2D(name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def ResNet_2(input_shape):

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64], stage=2, block='a')
    X = identity_block(X, f=3, filters=[64, 64], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='a')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256], stage=4, block='a')
    X = identity_block(X, 3, [256, 256], stage=4, block='b')
    X = identity_block(X, 3, [256, 256], stage=4, block='c')
    X = identity_block(X, 3, [256, 256], stage=4, block='d')
    X = identity_block(X, 3, [256, 256], stage=4, block='e')
    X = identity_block(X, 3, [256, 256], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512], stage=5, block='a')
    X = identity_block(X, 3, [512, 512], stage=5, block='b')
    X = identity_block(X, 3, [512, 512], stage=5, block='c')

    if X.shape[2] != 1:
        X = tf.keras.layers.AveragePooling2D(name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(2, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model