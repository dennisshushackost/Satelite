from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation

class UNET:
    """
    This class defines a UNET architecture for semantic segmentation 
    of satellite images.
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
    
    def encoder_block(self, input, num_filters):
        """
        Encoder block for the UNET architecture
        """
        x = self.conv_block(input, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, input, skip_features, num_filters):
        """
        Decoder block for the UNET architecture
        """
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x
    
    def build_model(self):
        """
        Build the UNET model
        """
        inputs = Input(self.input_shape)
        
        # New encoder blocks with smaller number of filters
        s0, p0 = self.encoder_block(inputs, 16)
        s1, p1 = self.encoder_block(p0, 32)
        
        # Existing encoder blocks
        s2, p2 = self.encoder_block(p1, 64)
        s3, p3 = self.encoder_block(p2, 128)
        s4, p4 = self.encoder_block(p3, 256)
        s5, p5 = self.encoder_block(p4, 512)
        
        # Bridge
        b1 = self.conv_block(p5, 1024)
        
        # Existing decoder blocks
        d5 = self.decoder_block(b1, s5, 512)
        d4 = self.decoder_block(d5, s4, 256)
        d3 = self.decoder_block(d4, s3, 128)
        d2 = self.decoder_block(d3, s2, 64)
        
        # New decoder blocks corresponding to the added encoder blocks
        d1 = self.decoder_block(d2, s1, 32)
        d0 = self.decoder_block(d1, s0, 16)
        
        # Output layer
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d0)
        
        model = Model(inputs, outputs, name='UNET')
        return model
