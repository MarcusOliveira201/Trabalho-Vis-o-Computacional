import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential

def convolution_operation(entered_input, filters=64):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    
    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)
    
    return act2

def encoder(entered_input, filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    enc1 = convolution_operation(entered_input, filters)
    MaxPool1 = MaxPooling2D(strides = (2,2))(enc1)
    return enc1, MaxPool1

def decoder(entered_input, skip, filters=64):
    # Upsampling and concatenating the essential features
    Upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
    Connect_Skip = Concatenate()([Upsample, skip])
    out = convolution_operation(Connect_Skip, filters)
    return out

def U_NetBase(image_size, num_classes, activation):
  # Take the image size and shape
  input_shape = (image_size, image_size, 3)
  input1 = Input(input_shape)
  
  # Construct the encoder blocks
  skip1, encoder_1 = encoder(input1, 64)
  skip2, encoder_2 = encoder(encoder_1, 64*2)
  skip3, encoder_3 = encoder(encoder_2, 64*4)
  skip4, encoder_4 = encoder(encoder_3, 64*8)
  
  # Preparing the next block
  conv_block = convolution_operation(encoder_4, 64*16)
  
  # Construct the decoder blocks
  decoder_1 = decoder(conv_block, skip4, 64*8)
  decoder_2 = decoder(decoder_1, skip3, 64*4)
  decoder_3 = decoder(decoder_2, skip2, 64*2)
  decoder_4 = decoder(decoder_3, skip1, 64)
  
  out = Conv2D(num_classes, 1, padding="same", activation=activation)(decoder_4)

  model = Model(input1, out)
  return model

def main():
    input_shape = (256, 256, 3)
    model = U_NetBase(input_shape)
    model.summary()
    tf.keras.utils.plot_model(model, "model.png", show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

if __name__ == "__main__":
    main()