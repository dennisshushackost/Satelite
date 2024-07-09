"""
Attention U-net:
https://arxiv.org/pdf/1804.03999.pdf

Recurrent residual Unet (R2U-Net) paper
https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
(Check fig 4.)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    """
    This is a convolutional block with Batch Normalization and ReLU activation.
    """
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def unet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    This is a UNET model with 5 downsampling and 5 upsampling layers.
    '''
    # network structure
    FILTER_NUM = 32 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    
    # DownRes 1, convolution + pooling
    conv_256 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2,2))(conv_256)
    # DownRes 2
    conv_128 = conv_block(pool_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 3
    conv_64 = conv_block(pool_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 4
    conv_32 = conv_block(pool_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 5, convolution only
    conv_16 = conv_block(pool_16, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    # Upsampling layers
    # UpRes 6
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    up_256 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_128)
    up_256 = layers.concatenate([up_256, conv_256], axis=3)
    up_conv_256 = conv_block(up_256, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 10
    up_512 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_256)
    up_conv_512 = conv_block(up_512, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 11
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_512)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel
    # Model 
    model = models.Model(inputs, conv_final, name="UNet")
    print(model.summary())
    return model

def resunet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    """
    Residual U-Net architecture with 5 downsampling and 5 upsampling layers.
    
    This model combines the ideas of U-Net and ResNet, using residual blocks
    and skip connections to improve gradient flow and feature propagation.
    
    Args:
        input_shape (tuple): Shape of the input image.
        NUM_CLASSES (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        batch_norm (bool): Whether to use batch normalization.
    
    Returns:
        tf.keras.Model: The compiled Residual U-Net model.
    """
    FILTER_NUM = 32
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2

    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    # Encoder (downsampling) path
    conv_256 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)
    
    conv_128 = res_conv_block(pool_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    # Bridge
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    
    # Decoder (upsampling) path
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    up_256 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_128)
    up_256 = layers.concatenate([up_256, conv_256], axis=3)
    up_conv_256 = res_conv_block(up_256, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    # Final upsampling and output
    up_512 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_256)
    up_conv_512 = res_conv_block(up_512, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_512)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs, conv_final, name="ResUNet")
    print(model.summary())
    return model


def attunet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    """
    Attention U-Net architecture with 5 downsampling and 5 upsampling layers.
    
    This model incorporates attention mechanisms to focus on relevant features
    during the upsampling process, improving the network's ability to capture
    fine details.
    
    Args:
        input_shape (tuple): Shape of the input image.
        NUM_CLASSES (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        batch_norm (bool): Whether to use batch normalization.
    
    Returns:
        tf.keras.Model: The compiled Attention U-Net model.
    """
    FILTER_NUM = 32
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2

    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    # Encoder (downsampling) path
    conv_256 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)
    
    conv_128 = conv_block(pool_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    conv_64 = conv_block(pool_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    conv_32 = conv_block(pool_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    # Bridge
    conv_16 = conv_block(pool_16, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    
    # Decoder (upsampling) path
    gating_32 = gating_signal(conv_16, 8*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 8*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    gating_64 = gating_signal(up_conv_32, 4*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 4*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    gating_128 = gating_signal(up_conv_64, 2*FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, 2*FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    gating_256 = gating_signal(up_conv_128, FILTER_NUM, batch_norm)
    att_256 = attention_block(conv_256, gating_256, FILTER_NUM)
    up_256 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_128)
    up_256 = layers.concatenate([up_256, att_256], axis=3)
    up_conv_256 = conv_block(up_256, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    # Final upsampling and output
    up_512 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_256)
    up_conv_512 = conv_block(up_512, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_512)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs, conv_final, name="Attention_UNet")
    print(model.summary())
    return model


if __name__ == "__main__":
    # Test the model architectures
    input_shape = (256, 256, 4)
    attunet(input_shape)

    