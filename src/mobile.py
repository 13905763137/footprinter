import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, ReLU, Conv2DTranspose, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model

def Encoder(input_shape=(128, 128, 3)):

    base_model = MobileNetV2(input_shape=input_shape, include_top=False)

    # Use the activations of these layers for the Skip connection
    layer_names = [
     'block_1_expand_relu',   # OUTPUT_SHAPE: (BS, 64, 64, 96)
     'block_3_expand_relu',   # OUTPUT_SHAPE: (BS, 32, 32, 144)
     'block_6_expand_relu',   # OUTPUT_SHAPE: (BS, 16, 16, 192)
     'block_13_expand_relu',  # OUTPUT_SHAPE: (BS, 8, 8, 576)
     'block_16_project'       # OUTPUT_SHAPE: (BS, 4, 4, 320)
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction encoder with 5 outputs
    # The last output is the input of the decoder
    # the 4th, 3rd, 2nd, and 1st outputs are the 1st, 2nd, 3rd, and 4th skip connections the decoder
    down_stack = Model(inputs=base_model.input, outputs=layers)
   
    # Make it non-trainable
    down_stack.trainable = False
   
    return down_stack 

def upsampler_block(nfilters, size=3, strides=2, norm_type='batchnorm', apply_dropout=False):
 
    """
    source: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L220
    Upsamples an input with a Conv2DTranspose followed by Batchnorm, Dropout and Relu activation
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args: nfilters: number of filters
          size: filter size
          norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
          apply_dropout: If True, adds the dropout layer
    Returns:
          An upsampler Sequential Model : (nrows, ncols) --> (new_nrows, new_ncols)
                                          new_nrows = (nrows - 1) * strides[0] + size[0]
                                          new_ncols = (ncols - 1) * strides[1] + size[1]
                                          See: https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2DTranspose(nfilters, size, strides=2,
                               padding='same',
                               kernel_initializer = "he_normal",
                               use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))


    result.add(ReLU())

    return result


def unet_model(input_shape=(128, 128, 3)):

    decoder = [
               upsampler_block(512, 3),  # (BS, 4, 4, 320) -> (BS, 8, 8, 512) + (BS, 8, 8, 576) = (BS, 8, 8, 1088)
               upsampler_block(256, 3),  # (BS, 8, 8, 1088) -> (BS, 16, 16, 256) + (BS, 16, 16, 192) = (BS, 16, 16, 448)
               upsampler_block(128, 3),  # (BS, 16, 16, 448) -> (BS, 32, 32, 128) + (BS, 32, 32, 144) = (BS, 32, 32, 172)
               upsampler_block(64, 3)   # (BS, 32, 32, 172) -> (BS, 64, 64, 64) + (BS, 64, 64, 96) = (BS, 64, 64, 160)
               ]
    inputs = Input(shape=input_shape)
    encoder = Encoder(input_shape)
    encoder_outputs = encoder(inputs)
    #skip connections are ordered from last to first
    encoder_skips_backward, x = encoder_outputs[:-1], encoder_outputs[-1]
    #reorder the skip connection
    encoder_skips_forward = reversed(encoder_skips_backward)
    
    for upsampler, encoder_skip in zip(decoder, encoder_skips_forward):
        
        #Upsample with the unit upsampling block
        x = upsampler(x)
        #Concatenate the upsampled tensor with the skip connections
        x = Concatenate()([x, encoder_skip])
    
    #One last upsampling layer to predict the target binary mask: (BS, 64, 64, 160) -> (BS, 128, 128, 1)  
    outputs = Conv2DTranspose(1, 3, strides=2, padding="same", activation = "sigmoid")(x)
    unet_model = Model(inputs = inputs, outputs = outputs)

    return unet_model
