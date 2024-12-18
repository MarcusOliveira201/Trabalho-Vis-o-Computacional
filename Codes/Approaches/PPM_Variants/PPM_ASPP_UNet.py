"""
This is the structure of the HRNet-32, an implementation of the CVPR 2019 paper "Deep High-Resolution Representation
Learning for Human Pose Estimation" using TensorFlow.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 06, 2019

"""
import tensorflow as tf
from keras.utils import plot_model  
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D,ReLU,LeakyReLU,Add,GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential


# from PPM_Simple import decoder_withResNet
# from PPM_Simple import decoder_default
# from PPM_Simple import convolution_operation_decoder_Unet
import os
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
########################################################
############ Deeplab Architecture ######################
########################################################
def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False,):
    x = layers.Conv2D(num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def PPM_With_ASPP_Unet(image_size, num_classes,activation):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    ##################### PPM Encoder Call ####################
    skip_list = []
    skip_list, outputPPM = PPM_Enconder(model_input)   
    
    # # resnet50 = keras.applications.ResNet50(
    # #     weights="imagenet", include_top=False, input_tensor=model_input
    # # )
    # x = resnet50.get_layer("conv4_block6_2_relu").output
    # x = DilatedSpatialPyramidPooling(x)
    x = DilatedSpatialPyramidPooling(outputPPM)

    ########## Decoder Block UNet Shape ###########
    n_filtro = 48
    #Level 4 - #Upsample 2x2 -  #Concate Skip4 + Up1 - Conv3x3 - Conv3x3
    up1 = layers.UpSampling2D(size=2,interpolation="bilinear")(x)
    x = layers.Concatenate(axis=-1)([skip_list[3], up1])
    x = convolution_block( x, num_filters=n_filtro*8, kernel_size=3)
    out_lvl4 = convolution_block( x, num_filters=n_filtro*8, kernel_size=3)
    
    #Level 3 - #Upsample 2x2 -  #Concate Skip3 + Up2 - Conv3x3 - Conv3x3
    up2 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl4)
    x = layers.Concatenate(axis=-1)([skip_list[2], up2])
    x = convolution_block( x, num_filters=n_filtro*4, kernel_size=3)
    out_lvl3 = convolution_block( x, num_filters=n_filtro*4, kernel_size=3)
    
    #Level 2 - #Upsample 2x2 -  #Concate Skip2 + Up3 - Conv3x3 - Conv3x3
    up3 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl3)
    x = layers.Concatenate(axis=-1)([skip_list[1], up3])
    x = convolution_block( x, num_filters=n_filtro*2, kernel_size=3)
    out_lvl2 = convolution_block( x, num_filters=n_filtro*2, kernel_size=3)
   
    #Level 1 - #Upsample 2x2 -  #Concate Skip1 + Up4 - Conv3x3 - Conv3x3
    up4 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl2)
    x = layers.Concatenate(axis=-1)([skip_list[0], up4])
    x = convolution_block( x, num_filters=n_filtro, kernel_size=3)
    out_lvl1 = convolution_block( x, num_filters=n_filtro, kernel_size=3)
   
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation=activation)(out_lvl1)
    return keras.Model(inputs=model_input, outputs=model_output)
   

def convolution_operation_base_Unet(entered_input, filters=64):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    
    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)
    
    return act2

def convolution_operation_PPM(entered_input, filters=64, kernel=(1,1)):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = kernel, padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    return act1

def PPM(entered_input,filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    enc1 = convolution_operation_PPM(entered_input, filters, kernel=(1,1)) # 1
    enc2 = convolution_operation_PPM(entered_input, filters, kernel=(3,3)) # 1
    enc3 = convolution_operation_PPM(entered_input, filters, kernel=(5,5)) # 1
    enc4 = convolution_operation_PPM(entered_input, filters, kernel=(7,7)) # 1
    merged = keras.layers.concatenate([enc1,enc2,enc3,enc4], axis=-1)

    MaxPool1 = MaxPooling2D(strides = (2,2))(merged)
    return merged, MaxPool1

def PPM_Enconder(input):
    # Take the image size and shape
    input1 = input
    n_filtro = 48

    # Construct the encoder blocks 
    skip1, encoder_1 = PPM(input1, n_filtro)
    skip2, encoder_2 = PPM(encoder_1,  n_filtro*2)
    skip3, encoder_3 = PPM(encoder_2, n_filtro*4)
    skip4, encoder_4 = PPM(encoder_3, n_filtro*8)
        
    # Preparing the next block
    conv_block = convolution_operation_base_Unet(encoder_4,  n_filtro*16)
    
    return [skip1,skip2,skip3,skip4],conv_block


# def mytest():
#     ########################################################
#     ################### Define Model #######################
#     ########################################################
#     NUM_CLASSES = 3
#     IMAGE_SIZE = 256
    
    
#     model = PPM_ASPP_UNET(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation="softmax")
#     model.summary()
#     patha="G:\\Meu Drive\\!Doutorado_UFMA-UFPI\\!Codes\\PPM\\Revista\\Revista\\Propostas_Experimentos_PPM\\1 - Combine Pyramid Pooling UNet\\"
#     plot_model(model, to_file= patha + "model_plot_PPM_ASPP_UNet_Shape.png", show_shapes=True, show_layer_names=True)
   

# if __name__ == '__main__':
#     mytest()

