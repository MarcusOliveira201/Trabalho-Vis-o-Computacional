"""
Adaptação do modulo convulucional Conformer como extrator de características da arquitetura padrão UNET usando TensorFlow .

@ Author: Caio Falcão caioefalcao@nca.ufma.br

@ Date created: Abr 20, 2024

@ Date created: Abr 26, 2024

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
from keras.utils import plot_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
from morpholayers import *
from morpholayers.layers import Dilation2D, Erosion2D, Opening2D, Closing2D, Gradient2D
from tensorflow.keras import layers
from morpholayers.initializers import *

#from keras_flops import get_flops


import os
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'



########################################################
############### UNet Architecture ######################
########################################################

def convolution_block(block_input, num_filters=48, kernel_size=3, dilation_rate=1, padding="same", use_bias=False, pool_size=(2, 2)):
    x = layers.Conv2D(num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        activation='relu'
    )(block_input)
    x = layers.BatchNormalization()(x)
    return x

# se for alternar com uma convolução normal, multiplicar por 4 por causa da concatenação.
# Fazer a segmentação separada, um encoder e decoder para cada uma das classes. No caso eu somo as 3 losses
def merge_PPM(entered_input, filters=48):

    entered_input2 = Erosion2D(1,kernel_size=(5,5))(entered_input)

    out_1 = convolution_block(entered_input2, filters, 1)
    out_3 = convolution_block(entered_input2, filters, 3)
    out_7 = convolution_block(entered_input2, filters, 5)
    out_11 = convolution_block(entered_input2, filters, 7)

    out_1_se = se_block(out_1, filters)
    out_3_se = se_block(out_3, filters)
    out_7_se = se_block(out_7, filters)
    out_11_se = se_block(out_11, filters)

    # merged = layers.concatenate([out_1, out_3, out_7, out_11], axis=-1)

    merged = layers.concatenate([out_1_se, out_3_se, out_7_se, out_11_se], axis=-1)

    enc1 = convolution_block(merged, filters) 

    enc2 = layers.BatchNormalization()(enc1)
    relu = layers.ReLU()(enc2)
    next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(relu)   
    skip_connection = relu

    return skip_connection, next_layer


def UNET_PPM_MORPH(image_size, num_classes,activation):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    ##################### UNet Encoder Call ####################
    skip_list = []
    skip_list, x = UNet_Enconder(model_input)   
    
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

################################################################
############### UNET Architecture Encoder ######################
################################################################
def bottleneck_block(entered_input, filters=48):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    
    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)
    
    return act2
    
def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((1, 1, ch))(x)  # Reshape for broadcasting
    return Multiply()([in_block, x])

def UNet_Enconder(input):
    # Take the image size and shape
    input1 = input
    n_filtro = 48

    # Construct the encoder blocks 
    skip1, encoder_1 = merge_PPM(input1, n_filtro)
    skip2, encoder_2 = merge_PPM(encoder_1,n_filtro*2)
    skip3, encoder_3 = merge_PPM(encoder_2, n_filtro*4)
    skip4, encoder_4 =  merge_PPM(encoder_3,n_filtro*8)

    # Preparing the next block
    conv_block = bottleneck_block(encoder_4,  n_filtro*16)
    
    return [skip1,skip2,skip3,skip4],conv_block

def mytest():
    ########################################################
    ################### Define Model #######################
    ########################################################
    NUM_CLASSES = 3
    IMAGE_SIZE = 128
    
    
    model = UNET_PPM_MORPH(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation="softmax")
    model.summary()
    #tf.keras.utils.plot_model(model, "model.png", show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
    #patha="G:\\Meu Drive\\!Doutorado_UFMA-UFPI\\!Codes\\PPM\\Revista\\Revista\\Customizando_Bloco_PPM\\1 - Conformer Conv_UNet copy\\"
    plot_model(model, to_file= "model_plot_UNet_ConformerConv2.png", show_shapes=True, show_layer_names=True)

    model.save("PPM_model_Teste.h5")

if __name__ == '__main__':
    mytest()

