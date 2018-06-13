from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import warnings

from keras.models import Model
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
from tensorflow import Dimension as Dim

from keras.utils import plot_model

###
#	This is the script which generates the actual model.
#	The idea is to take the ResNext block and integrate FSM block with each resnext block
#	For generating the resnext block, we modified the code available on
#	https://github.com/titu1994/Keras-ResNeXt
#	according to our requirements and introduces new block called fsm.
#	While creating a layer of resnext blocks, we integrate it with 
#	fsm block, Then fsm block of ewach layer is concatenated to get
#	FS block in the last layer. This FS block is added with
#	final layer's output to get final representation of the image.
#	
###

def __get_fsm_block(x, final_output, block_number, channel_axis):
    fsm = GlobalAveragePooling2D(data_format='channels_last')(x)
    if block_number < 4:
        fsm = Dense(128)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(64)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(final_output)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
    elif block_number < 8:
        fsm = Dense(256)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(128)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(final_output)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
    elif block_number < 14:
        fsm = Dense(512)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(128)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(final_output)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
    elif block_number < 22:
        fsm = Dense(512)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(128)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
        fsm = Dense(final_output)(fsm)
        fsm = BatchNormalization(axis=channel_axis)(fsm)
        fsm = Activation('relu')(fsm)
    fsm = Activation('sigmoid')(fsm)  # this is needed before we multiply fsm with each block
    return fsm

def __multiply_fsm(x, fsm, cardinality, channel_axis):
    org_shape = x.get_shape().as_list()
    normal_shape = org_shape[1:]
    reshaped = normal_shape[:-1]
    last = normal_shape[-1]
    reshaped.append(x.get_shape().as_list()[-1] // cardinality)
    reshaped.append(cardinality)
    x = Reshape(reshaped)(x)
    x = multiply([x, fsm])
    x = Reshape(normal_shape)(x)
    return x


def MLFN(input_shape=None, depth=29, cardinality=8, width=64, weight_decay=5e-4,
         include_top=True, weights=None, input_tensor=None,
         pooling=None, classes=767, T_dimension=20):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = createMLFN(classes, img_input, include_top, depth, cardinality, width,
                          weight_decay, pooling, T_dimension=T_dimension)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mlfn')
    return model


def createMLFN (nb_classes, img_input, include_top, depth=29, cardinality=8, width=4,
                      weight_decay=5e-4, pooling=None, T_dimension=100):

    if type(depth) is list or type(depth) is tuple:
        N = list(depth)
    else:
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []
    fsm_list = [] # This is the list containing all fsm blocks that are created in each layer
    channel_axis = -1
    fsm_blockNumber = 1
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters

    x = __initial_conv_block(img_input, weight_decay)
    for i in range(N[0]):
		#Initial layer consisting of FSM block and resnect block	
        fsm = __get_fsm_block(x, cardinality, fsm_blockNumber, channel_axis)
        fsm_blockNumber = fsm_blockNumber + 1
        x = __bottleneck_block(x, fsm, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)
        fsm_list.append(fsm)

    N = N[1:]  
    filters_list = filters_list[1:] 
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            fsm = __get_fsm_block(x, cardinality, fsm_blockNumber, channel_axis)
            fsm_blockNumber = fsm_blockNumber + 1
            fsm_list.append(fsm)
            if i == 0:
                x = __bottleneck_block(x, fsm, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, fsm, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)
    fsm_merge = concatenate(fsm_list, axis=channel_axis)
    print(fsm_merge.shape, 'sfm merged shape')

    phi_S = Dense(T_dimension)(fsm_merge)
    phi_R = Dense(T_dimension)(x)
    print(phi_R.shape, 'PHI_R SHAPE')
    print(phi_S.shape, 'PHI_S SHAPE')
    x = add([phi_R, phi_S])

    tensor = tf.convert_to_tensor([0.5], tf.float32)
    x = Lambda(lambda a: a * 0.5, name="Rlayer")(x)
    if include_top:
        x = GlobalAveragePooling2D(name = 'avg_pool')(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x

def __initial_conv_block(input, weight_decay=5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x

#This block is modified to integrate the FSM block
def __bottleneck_block(input, fsm, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, fsm, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    #    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = __multiply_fsm(x, fsm, cardinality, channel_axis)
    x = add([init, x])
    x = Activation('relu')(x)  # do we need this layer TODO: harmeet
    return x


def __grouped_convolution_block(input, fsm, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        print(c, ' c valuex')
        print(x.shape, ' X shape')
        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)
    return x


if __name__ == '__main__':
    model = MLFN((256, 128, 3), depth=[3,4,6,3], cardinality=8, width=64, T_dimension=1000)
    model.summary()
    plot_model(model, to_file='mlnfvList.png', show_shapes=True)
