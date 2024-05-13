import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate


class UNET:
    """
    This class defines a normal UNET architecture for semantic segmentation 
    of the satellite images.
    """
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        
    def conv_block(self, input, num_filters):
        """
        Convolutional block for the UNET architecture
        """
        x = Conv2D(num_filters, 3, padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    # Encoder block
    def encoder_block(self, input, num_filters):
        """
        Encoder block for the UNET architecture
        """
        x = self.conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p
    
    # Decoder block
    def decoder_block(self, input, skip_features, num_filters):
        """
        Decoder block for the UNET architecture
        """
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x
    
    # Build UNET with input size (1024, 1024, 4)
    def build_model(self):
        """
        Build the UNET model
        """
        inputs = Input(self.input_shape)
        
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)
        
        b1 = self.conv_block(p4, 1024)
        
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
        
        model = Model(inputs, outputs, name='UNET')
        return model