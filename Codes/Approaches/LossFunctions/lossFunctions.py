from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#https://www.sciencedirect.com/science/article/pii/S0895611121001750


import keras.backend as K

def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 2#0.75
    return K.sum(K.pow((1-pt_1), gamma))

#######################
smooth=100

ALPHA = 0.5
BETA = 0.5
GAMMA = 1
num_class = 4
#Tensorflow / Keras
def FocalTverskyLoss_(y_true, y_pred, smooth=1e-6):
        
        if y_pred.shape[-1] <= 1:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #5.
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #3.
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)
        
        
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky
#Tensorflow / Keras
def TverskyLoss_(y_true, y_pred, smooth=1e-6):
    
        if y_pred.shape[-1] <= 1:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #5.
            y_pred = tf.keras.activations.sigmoid(y_pred)
            #y_true = y_true[:,:,:,0:1]
        elif y_pred.shape[-1] >= 2:
            alpha = 0.3
            beta = 0.7
            gamma = 4/3 #3.
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)
           
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

############# Other 
#Keras
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky
        

def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
# Tensorflow/Keras
from keras import backend as K


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss



def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0,1,2,3))
    union = tf.reduce_sum(y_true, axis=(0,1,2,3)) + tf.reduce_sum(y_pred, axis=(0,1,2,3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)







###################################
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

#Keras
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

def Combo_loss(targets, inputs, eps=1e-9):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    
    return combo
# def dice_coef(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef(y_true, y_pred)

# def iou(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred)
#     sum_ = K.sum(y_true + y_pred)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return jac

# def jac_distance(y_true, y_pred):
#     y_truef=K.flatten(y_true)
#     y_predf=K.flatten(y_pred)

#     return - iou(y_true, y_pred)

# def mean_iou_old(y_true, y_pred):
#     yt0 = y_true[:,:,:,0]
#     yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
#     inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#     union = tf.math.count_nonzero(tf.add(yt0, yp0))
#     iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
#     return iou

# def seg_metrics(y_true, y_pred, metric_name,
#     metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
#     """
#     Compute mean metrics of two segmentation masks, via Keras.

#     IoU(A,B) = |A & B| / (| A U B|)
#     Dice(A,B) = 2*|A & B| / (|A| + |B|)

#     Args:
#         y_true: true masks, one-hot encoded.
#         y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#         metric_name: metric to be computed, either 'iou' or 'dice'.
#         metric_type: one of 'standard' (default), 'soft', 'naive'.
#           In the standard version, y_pred is one-hot encoded and the mean
#           is taken only over classes that are present (in y_true or y_pred).
#           The 'soft' version of the metrics are computed without one-hot
#           encoding y_pred.
#           The 'naive' version return mean metrics where absent classes contribute
#           to the class mean as 1.0 (instead of being dropped from the mean).
#         drop_last = True: boolean flag to drop last class (usually reserved
#           for background class in semantic segmentation)
#         mean_per_class = False: return mean along batch axis for each class.
#         verbose = False: print intermediate results such as intersection, union
#           (as number of pixels).
#     Returns:
#         IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
#           in which case it returns the per-class metric, averaged over the batch.

#     Inputs are B*W*H*N tensors, with
#         B = batch size,
#         W = width,
#         H = height,
#         N = number of classes
#     """

#     flag_soft = (metric_type == 'soft')
#     flag_naive_mean = (metric_type == 'naive')

#     # always assume one or more classes
#     num_classes = K.shape(y_true)[-1]

#     if not flag_soft:
#         # get one-hot encoded masks from y_pred (true masks should already be one-hot)
#         y_pred = K.one_hot(K.argmax(y_pred), num_classes)
#         y_true = K.one_hot(K.argmax(y_true), num_classes)

#     # if already one-hot, could have skipped above command
#     # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')

#     # intersection and union shapes are batch_size * n_classes (values = area in pixels)
#     axes = (1,2) # W,H axes of each image
#     intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
#     mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
#     union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

#     smooth = .001
#     iou = (intersection + smooth) / (union + smooth)
#     dice = 2 * (intersection + smooth)/(mask_sum + smooth)

#     metric = {'iou': iou, 'dice': dice}[metric_name]

#     # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
#     mask =  K.cast(K.not_equal(union, 0), 'float32')

#     if drop_last:
#         metric = metric[:,:-1]
#         mask = mask[:,:-1]

#     if verbose:
#         print('intersection, union')
#         print(K.eval(intersection), K.eval(union))
#         print(K.eval(intersection/union))

#     # return mean metrics: remaining axes are (batch, classes)
#     if flag_naive_mean:
#         return K.mean(metric)

#     # take mean only over non-absent classes
#     class_count = K.sum(mask, axis=0)
#     non_zero = tf.greater(class_count, 0)
#     non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
#     non_zero_count = tf.boolean_mask(class_count, non_zero)

#     if verbose:
#         print('Counts of inputs with class present, metrics for non-absent classes')
#         print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))

#     return K.mean(non_zero_sum / non_zero_count)

# def mean_iou(y_true, y_pred, **kwargs):
#     """
#     Compute mean Intersection over Union of two segmentation masks, via Keras.

#     Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
#     """
#     return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)

# def mean_dice(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via Keras.

    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='dice', **kwargs)