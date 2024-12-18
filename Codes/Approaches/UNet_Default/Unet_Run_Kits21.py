"""
Title: Multiclass semantic segmentation using DeepLabV3+
Author: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/08/31
Last modified: 2021/09/1
Description: Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation.
"""
"""
## Introduction

Semantic segmentation, with the goal to assign semantic labels to every pixel in an image,
is an essential computer vision task. In this example, we implement
the **DeepLabV3+** model for multi-class semantic segmentation, a fully-convolutional
architecture that performs well on semantic segmentation benchmarks.

### References:

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
"""

"""
## Downloading the data

We will use the [Crowd Instance-level Human Parsing Dataset](https://arxiv.org/abs/1811.12596)
for training our model. The Crowd Instance-level Human Parsing (CIHP) dataset has 38,280 diverse human images.
Each image in CIHP is labeled with pixel-wise annotations for 20 categories, as well as instance-level identification.
This dataset can be used for the "human part segmentation" task.
"""

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import argmax
#sys.path.insert(1, '/home/caio_matos/Codes/DeepLabPPM/')
#from DeepLabPPM import DeeplabV3PlusPPM

sys.path.insert(1, '/home/caio_matos/Codes/DatasetCodes/')
from kitsDataset import Dataset
from kitsDataset import defineTrainTestValidation,get_preprocessing,setDatabasePath,setResultsPath,denormalize,visualize,createBackboneFolders,getSizeDataset,Dataloader

sys.path.insert(1, '/home/caio_matos/Codes/UNet_Default/')

from Unet_Model import U_NetBase
import segmentation_models as sm
sm.set_framework('tf.keras')

from segmentation_models.losses import CategoricalCELoss

"""shell
gdown https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz
unzip -q instance-level-human-parsing.zip
"""

"""
## Creating a TensorFlow Dataset

Training on the entire CIHP dataset with 38,280 images takes a lot of time, hence we will be using
a smaller subset of 200 images for training our model in this example.
"""

#IMAGE_SIZE = 256
IMAGE_SIZE = 512
BATCH_SIZE = 4
CLASSES = ['background','kidney','tumor','cyst']
#CLASSES = ['background','kidney']
EPOCHS = 50
#DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
#NUM_TRAIN_IMAGES = 1000
#NUM_VAL_IMAGES = 50


pathDataset = "/home/caio_matos/Dataset/kits21_Dataset/kits21/kits21/data/"
setDatabasePath(pathDataset)

resultsPath = "/home/caio_matos/Dataset/kits21_Dataset/kits21/kits21/Results/"
setResultsPath(resultsPath)

x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir = defineTrainTestValidation(70,20,10)

# train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
# train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
# val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
#     NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
# ]
# val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
#     NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
# ]


# def read_image(image_path, mask=False):
#     image = tf.io.read_file(image_path)
#     if mask:
#         image = tf.image.decode_png(image, channels=1)
#         image.set_shape([None, None, 1])
#         image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
#     else:
#         image = tf.image.decode_png(image, channels=3)
#         image.set_shape([None, None, 3])
#         image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
#         image = image / 127.5 - 1
#     return image


# def load_data(image_list, mask_list):
#     image = read_image(image_list)
#     mask = read_image(mask_list, mask=True)
#     return image, mask


# def data_generator(image_list, mask_list):
#     dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
#     dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#     return dataset


# train_dataset = data_generator(train_images, train_masks)
# val_dataset = data_generator(val_images, val_masks)

#------------- DATAGENERATOR TO KITS21 -------------#

preprocess_input = sm.get_preprocessing("resnet50")
## Train Dataset ##
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input),
    )

train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

############### #Verify image from Train Dataset
n = 10
ids = np.random.choice(np.arange(len(train_dataset)), size=n)

for i in ids:
    
    image, mask = train_dataset[i] # get some sample
    visualize(
        image=image, 
        kidney=mask[..., 1].squeeze(),
        tumor=mask[..., 2].squeeze(),
        cyst=mask[..., 3].squeeze(),
        background_mask=mask[..., 0].squeeze(),
        path="/home/caio_matos/Codes/UNet_Default/Outputs/Samples/",
        count=str(i) + "_Train"   
    )


########### Validation Dataset ##################
val_dataset = Dataset(
    x_valid_dir,
    y_valid_dir, 
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input),
    )

valid_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)

#NUM_CLASSES = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation

NUM_CLASSES = len(CLASSES)
# check shapes for errors
print(train_dataloader[0][0].shape)
print(train_dataloader[0][1].shape)
print((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES))
assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES)

#print("Train Dataset:", getSizeDataset(train_dataset))
#print("Val Dataset:", getSizeDataset(val_dataset))





########################################################
############ Deeplab Architecture ######################
########################################################
# def convolution_block(
#     block_input,
#     num_filters=256,
#     kernel_size=3,
#     dilation_rate=1,
#     padding="same",
#     use_bias=False,
# ):
#     x = layers.Conv2D(
#         num_filters,
#         kernel_size=kernel_size,
#         dilation_rate=dilation_rate,
#         padding="same",
#         use_bias=use_bias,
#         kernel_initializer=keras.initializers.HeNormal(),
#     )(block_input)
#     x = layers.BatchNormalization()(x)
#     return tf.nn.relu(x)
# def DilatedSpatialPyramidPooling(dspp_input):
#     dims = dspp_input.shape
#     x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
#     x = convolution_block(x, kernel_size=1, use_bias=True)
#     out_pool = layers.UpSampling2D(
#         size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
#         interpolation="bilinear",
#     )(x)

#     out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
#     out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
#     out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
#     out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

#     x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
#     output = convolution_block(x, kernel_size=1)
#     return output

# def DeeplabV3Plus(image_size, num_classes):
#     model_input = keras.Input(shape=(image_size, image_size, 3))
#     resnet50 = keras.applications.ResNet50(
#         weights="imagenet", include_top=False, input_tensor=model_input
#     )
#     x = resnet50.get_layer("conv4_block6_2_relu").output
#     x = DilatedSpatialPyramidPooling(x)

#     input_a = layers.UpSampling2D(
#         size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
#         interpolation="bilinear",
#     )(x)
#     input_b = resnet50.get_layer("conv2_block3_2_relu").output
#     input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

#     x = layers.Concatenate(axis=-1)([input_a, input_b])
#     x = convolution_block(x)
#     x = convolution_block(x)
#     x = layers.UpSampling2D(
#         size=(image_size // x.shape[1], image_size // x.shape[2]),
#         interpolation="bilinear",
#     )(x)
#     model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
#     return keras.Model(inputs=model_input, outputs=model_output)

########################################################
################### Define Model #######################
########################################################

#model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
activation = 'softmax'
#model = DeeplabV3PlusPPM(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=activation)
model = U_NetBase(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation=activation)

#model=deeplabv3_plus(num_classes=NUM_CLASSES)
#model.summary()

#Loss Function



jaccard_loss = sm.losses.JaccardLoss()
#loss = CategoricalCELoss()
dice_loss = sm.losses.DiceLoss() 
#dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 0.5])) 
focal_loss = sm.losses.BinaryFocalLoss() if NUM_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
total_loss = jaccard_loss + (1 * focal_loss)

#Metrics
iou_default = sm.metrics.IOUScore(threshold=0.5)
iou_bg = sm.metrics.IOUScore( threshold=0.5,class_indexes=0, name="iou_BG")
iou_kidney = sm.metrics.IOUScore( threshold=0.5,class_indexes=1, name="iou_Kidney")
iou_tumor = sm.metrics.IOUScore( threshold=0.5,class_indexes=2, name="iou_Tumor")
iou_cyst = sm.metrics.IOUScore( threshold=0.5,class_indexes=3, name="iou_Cyst")

fscore_default = sm.metrics.FScore(threshold=0.5)
fscore_bg = sm.metrics.FScore( threshold=0.5,class_indexes=0, name="fscore_BG")
fscore_kidney = sm.metrics.FScore( threshold=0.5,class_indexes=1, name="fscore_Kidney")
fscore_tumor = sm.metrics.FScore( threshold=0.5,class_indexes=2, name="fscore_Tumor")
fscore_cyst = sm.metrics.FScore( threshold=0.5,class_indexes=3, name="fscore_Cyst")

# metrics = [iou_default, iou_bg, iou_kidney,
#             fscore_default, fscore_bg, fscore_kidney]

metrics = [iou_default, iou_bg, iou_kidney, iou_tumor,iou_cyst,
            fscore_default, fscore_bg, fscore_kidney,fscore_tumor,fscore_cyst]

#metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=total_loss,
    metrics=metrics,
)  
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     loss=total_loss,
#     #metrics=metrics,
#     metrics=[sm.metrics.IOUScore( threshold=0.5),sm.metrics.FScore( threshold=0.5)],
# )
#pathOutputModel = "/home/caio_matos/Codes/DeepLab/Results/Model/" 
callbacks = [
    keras.callbacks.ModelCheckpoint('/home/caio_matos/Codes/UNet_Default/Outputs/Models/best_model'+'_UNET_D_'+str(EPOCHS) +'.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.EarlyStopping(patience=8),
    keras.callbacks.ReduceLROnPlateau(),
]
# filename = './best_model_DeepLabv3+.h5'
# callbacks = [
#         keras.callbacks.ModelCheckpoint(filepath = pathOutputModel+filename, save_weights_only=True, save_best_only=True, mode='min'),
#         keras.callbacks.EarlyStopping(patience=16),
#         keras.callbacks.ReduceLROnPlateau(),
#     ]
########################################################
################### Train Step #########################
########################################################
#history = model.fit(train_dataloader, validation_data=valid_dataloader, epochs=25)

history = model.fit(
            train_dataloader,
            steps_per_epoch=len(train_dataloader), 
            validation_data=valid_dataloader,
            validation_steps=len(valid_dataloader),
            epochs=EPOCHS,
            callbacks=callbacks)

            
pathOutputEvaluationTrain = "/home/caio_matos/Codes/UNet_Default/Outputs/TrainGraphics/"
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(pathOutputEvaluationTrain+'TrainPlot'+'.png')
plt.show()


########################################################
################### Testing Model ######################
########################################################

pathOutputScoreTest = "/home/caio_matos/Codes/UNet_Default/Outputs/TestScore/" + "UnetDefault_50_512"
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input),
)
test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

# load best weights
model.load_weights('/home/caio_matos/Codes/UNet_Default/Outputs/Models/best_model'+'_UNET_D_'+str(EPOCHS) +'.h5') 

################################################# Teste Evaluate Model ######################################
fileScore = open(pathOutputScoreTest + ".txt", "w")
fileScore.write("#####################\n")
fileScore.write("Result Segmentation "+"\n")
fileScore.write("#####################\n")
fileScore.write("#####################\n")

scores = model.evaluate(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
fileScore.write("Loss: {:.5}".format(scores[0])+"\n")
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))
    fileScore.write("mean {}: {:.5}".format(metric.__name__, value))
    fileScore.write("\n")
fileScore.close()    

########################################################
################### Sample Predictions #################
########################################################
pathOutputEvaluationTest = "/home/caio_matos/Codes/UNet_Default/Outputs/ImagesTest/"
n = 100
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    print("Image number: ",i)
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    pr_mask = model.predict(image).round()
    
    # pred = np.argmax(pr_mask, axis = 1)
    # label = np.argmax(gt_mask, axis = 1)
    # print("########Predict#######")
    # print(pred) 
    # print("########Label#######")
    # print(label)

    visualize(
        image=denormalize(image.squeeze()),
        gt_maskBackground=gt_mask[..., 0].squeeze(),
        pr_maskBackground=pr_mask[..., 0].squeeze(),
        gt_maskKidney=gt_mask[..., 1].squeeze(),
        pr_maskKidney=pr_mask[..., 1].squeeze(),
        gt_maskTumor=gt_mask[..., 2].squeeze(),
        pr_maskTumor=pr_mask[..., 2].squeeze(),
        gt_maskCyst=gt_mask[..., 3].squeeze(),
        pr_maskCyst=pr_mask[..., 3].squeeze(),
        path=pathOutputEvaluationTest,
        count = i
    )