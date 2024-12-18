from glob import glob
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model

# import os
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
########################################################
############ UNet + PPM Architecture ###################
########################################################


def PPM_Model(image_size, num_classes, activation):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    ##################### PPM Encoder Call ####################
    skipList, outputEnconderPPM = PPM_Enconder(model_input)   
    
    ##################### PPM Dencoder Call ####################
    outputDecoderPPM = PPM_Decoder(skipList, outputEnconderPPM)
    
    model_output = layers.Conv2D(num_classes , kernel_size=(1, 1), padding="same", activation=activation)(outputDecoderPPM)
    return keras.Model(inputs=model_input, outputs=model_output)

############# Enconder and Decoder Blocks ##################

def PPM_Enconder(input):
    # Take the image size and shape
    input1 = input
    n_filtro = 48

    # Construct the encoder blocks 
    skip1, encoder_1 = PPM_Blocks(input1, n_filtro)
    skip2, encoder_2 = PPM_Blocks(encoder_1,  n_filtro*2)
    skip3, encoder_3 = PPM_Blocks(encoder_2, n_filtro*4)
    skip4, encoder_4 = PPM_Blocks(encoder_3, n_filtro*8)
        
    # Preparing the next block
    conv_block = convolution_operation_base_Unet(encoder_4,  n_filtro*16)
    skipList = [skip1,skip2,skip3,skip4]
    return skipList,conv_block

def PPM_Decoder(skipsList, conv_block):
    
    n_filtro = 48
    # Construct the dencoder blocks 
    decoder_1 = Skip_Connections(conv_block, skipsList[3], n_filtro*8)
    decoder_2 = Skip_Connections(decoder_1, skipsList[2], n_filtro*4)
    decoder_3 = Skip_Connections(decoder_2, skipsList[1],  n_filtro*2)
    decoder_4 = Skip_Connections(decoder_3, skipsList[0],  n_filtro)
            
    return decoder_4
    
def PPM_Blocks(entered_input,filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    enc1 = convolution_operation_PPM(entered_input, filters, kernel=(1,1)) # 1
    enc2 = convolution_operation_PPM(entered_input, filters, kernel=(3,3)) # 1
    enc3 = convolution_operation_PPM(entered_input, filters, kernel=(5,5)) # 1
    enc4 = convolution_operation_PPM(entered_input, filters, kernel=(7,7)) # 1
    merged = keras.layers.concatenate([enc1,enc2,enc3,enc4], axis=-1)

    MaxPool1 = MaxPooling2D(strides = (2,2))(merged)
    return merged, MaxPool1

############# Conv Blocks ##################
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

def Skip_Connections(entered_input, skip, filters=64):
    # Upsampling and concatenating the essential features
    Upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
    Connect_Skip = Concatenate()([Upsample, skip])
    out = convolution_operation_base_Unet(Connect_Skip, filters)
    return out



# ########################################################
# ################### Define Model #######################
# ########################################################
# NUM_CLASSES = 3
# IMAGE_SIZE = 512
# Act="softmax"
# model = PPM_Model(IMAGE_SIZE,NUM_CLASSES,Act)
# model.summary()
# patha="G:\\Meu Drive\\!Doutorado_UFMA-UFPI\\!Codes\\PPM\\Revista\\Revista\\ModelImage\\"
# plot_model(model, to_file= patha + "model_plot_PPM+dEFAULT.png", show_shapes=True, show_layer_names=True)