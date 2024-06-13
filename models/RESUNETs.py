from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Multiply

class AttentionUNet:
    """
    Attention U-Net architecture for semantic segmentation of satellite images.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def conv_block(self, inputs, num_filters):
        """Basic convolutional block with Batch Normalization and ReLU activation."""
        x = Conv2D(num_filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def attention_block(self, inputs, skip_features, num_filters):
        """Attention block with skip connections."""
        g = Conv2D(num_filters, 1, padding='same')(inputs)
        g = BatchNormalization()(g)

        x = Conv2D(num_filters, 1, padding='same')(skip_features)
        x = BatchNormalization()(x)

        psi = Activation('relu')(Concatenate()([g, x]))
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = Activation('sigmoid')(psi)

        return Multiply()([skip_features, psi])

    def encoder_block(self, inputs, num_filters):
        """Encoder block with MaxPooling."""
        x = self.conv_block(inputs, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        """Decoder block with Transposed Convolution and attention block."""
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
        x = self.attention_block(x, skip_features, num_filters)
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self):
        """Construct the Attention U-Net model."""
        inputs = Input(self.input_shape)

        # Encoder path
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        # Bridge
        b = self.conv_block(p4, 1024)

        # Decoder path with attention blocks
        d4 = self.decoder_block(b, s4, 512)
        d3 = self.decoder_block(d4, s3, 256)
        d2 = self.decoder_block(d3, s2, 128)
        d1 = self.decoder_block(d2, s1, 64)

        # Additional upsampling layers
        u1 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(d1)
        u1 = self.conv_block(u1, 32)

        u2 = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(u1)
        u2 = self.conv_block(u2, 16)

        # Output layer
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u2)

        return Model(inputs, outputs, name='AttentionUNet')
