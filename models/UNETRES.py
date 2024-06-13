from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Add

class RESIDUALUNET:
    """
    U-Net architecture with residual blocks for semantic segmentation of satellite images.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def res_conv_block(self, inputs, num_filters):
        """Residual convolutional block with Batch Normalization and ReLU activation."""
        x = Conv2D(num_filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        shortcut = Conv2D(num_filters, 1, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
        
        res_path = Add()([x, shortcut])
        res_path = Activation('relu')(res_path)
                
        return res_path

    def encoder_block(self, inputs, num_filters):
        """Encoder block with MaxPooling and residual connection."""
        x = self.res_conv_block(inputs, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        """Decoder block with Transposed Convolution, residual connection, and skip connections."""
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
        x = Concatenate()([x, skip_features])
        x = self.res_conv_block(x, num_filters)
        return x

    def build_model(self):
        """Construct the U-Net model."""
        inputs = Input(self.input_shape)

        # Encoder path
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        # Bridge
        b = self.res_conv_block(p4, 1024)

        # Decoder path
        d4 = self.decoder_block(b, s4, 512)
        d3 = self.decoder_block(d4, s3, 256)
        d2 = self.decoder_block(d3, s2, 128)
        d1 = self.decoder_block(d2, s1, 64)

        # Additional upsampling layers
        u1 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(d1)
        u1 = self.res_conv_block(u1, 32)

        u2 = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(u1)
        u2 = self.res_conv_block(u2, 16)

        # Output layer
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u2)

        return Model(inputs, outputs, name='UNET')

# Example usage
model = RESIDUALUNET((256, 256, 4)).model
model.summary()
