import tensorflow as tf
from keras.utils import plot_model  
import numpy as np
import os
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,UpSampling2D, Activation, ReLU,LeakyReLU,Add,GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'

########################################################
############ PPM Default Architecture ######################
########################################################
def conv_PPM(entered_input, filters=64, kernel=(1,1)):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = kernel, padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    return act1
def upsampling_PPM(x,image_size):
    input_a = layers.UpSampling2D(
        size=(image_size[-3] // x.shape[1], image_size[-2] //  x.shape[2]),
        interpolation="bilinear",
    )(x)
    return input_a
def PPM_Block(entered_input,filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    MaxPool1 = MaxPooling2D(strides = (2,2))(entered_input)
    enc1 = upsampling_PPM(conv_PPM(MaxPool1, filters, kernel=(1,1)),entered_input.shape) 
    enc2 = upsampling_PPM(conv_PPM(MaxPool1, filters, kernel=(3,3)),entered_input.shape)  
    enc3 = upsampling_PPM(conv_PPM(MaxPool1, filters, kernel=(5,5)),entered_input.shape)  
    enc4 = upsampling_PPM(conv_PPM(MaxPool1, filters, kernel=(7,7)),entered_input.shape)  
    merged = keras.layers.concatenate([entered_input,enc1,enc2,enc3,enc4], axis=-1)
   
    return merged
def PPM_Net(image_size, num_classes, activation):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    # Take the image size and shape
    input1 = model_input
    n_filtro = 48

    # Construct the encoder blocks 
    merged = PPM_Block(input1, n_filtro)
   
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation=activation)(merged)
    return keras.Model(inputs=model_input, outputs=model_output)
####################################################################
############ New PPM Approach ######################
#https://medium.com/analytics-vidhya/semantic-segmentation-in-pspnet-with-implementation-in-keras-4843d05fc025
########################################################

def conv_block(X,filters,block):
    # resiudal block with dilated convolutions
    # add skip connection at last after doing convoluion operation to input X
    
    b = 'block_'+str(block)+'_'
    f1,f2,f3 = filters
    X_skip = X
    # block_a
    X = Conv2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
                      padding='same',kernel_initializer='he_normal',name=b+'a')(X)
    X = BatchNormalization(name=b+'batch_norm_a')(X)
    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)
    # block_b
    X = Conv2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
                      padding='same',kernel_initializer='he_normal',name=b+'b')(X)
    X = BatchNormalization(name=b+'batch_norm_b')(X)
    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)
    # block_c
    X = Conv2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
                      padding='same',kernel_initializer='he_normal',name=b+'c')(X)
    X = BatchNormalization(name=b+'batch_norm_c')(X)
    # skip_conv
    X_skip = Conv2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b+'add')([X,X_skip])
    X = ReLU(name=b+'relu')(X)
    return X
    
def base_feature_maps(input_layer):
    # base covolution module to get input image feature maps 
    
    # block_1
    base = conv_block(input_layer,[32,32,64],'1')
    # block_2
    base = conv_block(base,[64,64,128],'2')
    # block_3
    base = conv_block(base,[128,128,256],'3')
    return base

def pyramid_feature_maps(input_layer):
    # pyramid pooling module
    
    base = base_feature_maps(input_layer)
    # red
    red = GlobalAveragePooling2D(name='red_pool')(base)
    red = tf.keras.layers.Reshape((1,1,256))(red)
    red = Conv2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(red)
    red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(red)
    # yellow
    yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
    yellow = Conv2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(yellow)
    yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)
    # blue
    blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
    blue = Conv2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)
    # green
    green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
    green = Conv2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(green)
    green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base,red,yellow,blue,green])

def PPM_Last_Model(image_size, num_classes, activation):
    input_layer = keras.Input(shape=(image_size, image_size, 3))
    X = pyramid_feature_maps(input_layer)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation=activation)(X)
    # X = Conv2D(num_classes,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)
    # X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    # X = Activation('sigmoid',name='last_conv_relu')(X)
    # X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)
    return keras.Model(inputs=input_layer, outputs=model_output)
    


# def mytest():
#     ########################################################
#     ################### Define Model #######################
#     ########################################################
#     NUM_CLASSES = 3
#     IMAGE_SIZE = 256
    
#     model = last_conv_module(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation="softmax")
#     model.summary()
#     patha="G:\\Meu Drive\\!Doutorado_UFMA-UFPI\\!Codes\\PPM\\Revista\\Revista\\PropostasExperimentos\\0 - PPM - Padrao\\Model_Plot\\"
#     plot_model(model, to_file= patha + "model_plot_PPM_Padrão_v2_170124.png", show_shapes=True, show_layer_names=True)
   

# if __name__ == '__main__':
#     mytest()