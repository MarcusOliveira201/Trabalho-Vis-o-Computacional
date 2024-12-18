import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from time import time
import tensorflow as tf
# import albumentations as A
import matplotlib.pyplot as plt
#####################################
import sys
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,concatenate, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, BatchNormalization, Activation,Input, add,multiply,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN
import tensorflow.keras.backend as K

sys.path.insert(1, "C://Users//suporte//Desktop//Mestrado//caio_falcao_doc//Codes//Approaches//LossFunctions//")
from lossFunctions import TverskyLoss
#####################################
import segmentation_models as sm
sm.set_framework('tf.keras')
#####################################


import os
import os
import numpy as np
from glob import glob
# from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import argmax

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# import keras_cv


#Dataset Imports
sys.path.insert(1, "C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/DatasetCodes/")
from kitsDataset import createBackboneFolders2,defineTrainTestValidation, visualize, Dataset,setResultsPath, setDatabasePath, Dataloader, get_preprocessing, denormalize

sys.path.insert(1, '/home/viplab/Documents/GitHub/caio_falcao_doc/Codes/Approaches/LossFunctions/')
from lossFunctions import FocalTverskyLoss,TverskyLoss,weighted_categorical_crossentropy,dsc,tversky_loss,dice_coef,focal_tversky_loss,class_tversky
#from segmentation_models.losses import CategoricalCELoss


sys.path.insert(1, 'C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/CentralCode/utils')
from utils import getApproach

#APPROACH_LIST = ["DeepLabv3_plus_Default","DeepLabv3_plus_PPM","UNet_Default","UNet_PPM"]
APPROACH_LIST = ["PPM_SE"]
#APPROACH_LIST = ["Unet_Model"]
#PREPROCESSING_LIST = ["Contrast Stretching","CLAHE","HE",]
IMAGE_SIZE = 128#256
BATCH_SIZE = 8 #4
CLASSES = ['background','kidney','tumor','cyst']
#CLASSES = ['background','kidney','tumor']
#CLASSES = ['kidney','tumor','cyst']
N_CLASSES = len(CLASSES)
CHANNEL  =  3
LR = 0.0001
EPOCHS = 1
ACTIVATION ='softmax'

print("######### Hyperparameters #########")
print("MODEL_CLASSES:", CLASSES)
print("BATCH_SIZE:", BATCH_SIZE)
print("N_CLASSES:", N_CLASSES)
print("EPOCHS:", EPOCHS)
print("APPROACH_LIST:",APPROACH_LIST)

pathDataset = "C:/Users/suporte/Desktop/Dataset/kits21_Dataset/kits21/kits21/data/"
# pathDataset = "C:/Users/suporte/Desktop/Dataset/kits23_Dataset/kits23/kits23/dataset/"
setDatabasePath(pathDataset)

resultsPath = "D:/Results/"
setResultsPath(resultsPath)


       
for APPROACH_NAME in APPROACH_LIST:

    ############ RUNS Experiments ######################
    RUNS_Size = 2
    for RUN in range(1,RUNS_Size):
        
        #Define Train Test and Validation from kits19 dataset
        x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir = defineTrainTestValidation(70,20,10)

        
        ################################################################################
        # Create Datasets - Train, Test and Validation
        ################################################################################
        # preprocess_input = sm.get_preprocessing("resnet50")
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir,
            classes=CLASSES 
            #augmentation=get_training_augmentation(),
            #preprocessing=get_preprocessing(preprocess_input)
        )

        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir,
            classes=CLASSES 
            #augmentation=get_validation_augmentation(),
            #reprocessing=get_preprocessing(preprocess_input)
        )

        # Dataset for validation images
        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir,
            classes=CLASSES 
            #augmentation=get_validation_augmentation(),
            #preprocessing=get_preprocessing(preprocess_input)
        )
        train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)  

        # check shapes for errors

        # print(train_dataloader[0][0].shape)
        # print(train_dataloader[0][1].shape)
    
        # print((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CLASSES))
        assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CLASSES)
        ################################################################################
        # Create the Model
        ################################################################################
        #Create struture directory backbones
        backboneGeneralPath = createBackboneFolders2(APPROACH_NAME + "_" + str(RUN))
        ####
        dataset = Dataset(x_test_dir, y_test_dir, classes=CLASSES)
        # n = 10
        # ids = np.random.choice(np.arange(len(dataset)), size=n)
        # for i in ids:
            
        #     image, mask = dataset[i] # get some sample
        #     visualize(
        #         image=image, 
        #         background_mask=mask[..., 0].squeeze(),
        #         kidney=mask[..., 1].squeeze(),
        #         tumor=mask[..., 2].squeeze(),
        #         cyst=mask[..., 3].squeeze(),
        #         path=backboneGeneralPath +"/Samples/",
        #         count=i    
        #     )
        ####

        print("Run Execution:", str(RUN))
        print("Number of Classes: ",N_CLASSES)
        print("Activation: ",ACTIVATION)
        print("Approach: ",APPROACH_NAME + "_" + str(RUN))
        
        
        model = getApproach(APPROACH_NAME,IMAGE_SIZE,N_CLASSES,ACTIVATION)
        

        ################################################################################
        # Define Loss Function, Metrics and Callbacks
        ################################################################################
        # define optomizer

        # optim = keras.optimizers.Adam(LR)
        optim = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
        # class_weights = [1e-6, 1.0, 1.0, 1.0]
        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        #jaccard_loss = sm.losses.JaccardLoss(class_indexes = [0, 1, 2, 3],class_weights=class_weights)
        focal_loss = sm.losses.BinaryFocalLoss() if N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
        # total_loss = focal_tversky_loss
        #focal_loss =  sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        #total_loss = jaccard_loss

        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        #total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
        #Metrics
        iou_default = sm.metrics.IOUScore(  threshold=0.5)
        iou_bg = sm.metrics.IOUScore(       threshold=0.5,  class_indexes=0, name="iou_BG")
        iou_kidney = sm.metrics.IOUScore(   threshold=0.5,  class_indexes=1, name="iou_Kidney")
        iou_tumor = sm.metrics.IOUScore(    threshold=0.5,  class_indexes=2, name="iou_Tumor")
        iou_cyst = sm.metrics.IOUScore(     threshold=0.5,  class_indexes=3, name="iou_Cyst")

        fscore_default = sm.metrics.FScore( threshold=0.5)
        fscore_bg = sm.metrics.FScore(      threshold=0.5,  class_indexes=0, name="fscore_BG")
        fscore_kidney = sm.metrics.FScore(  threshold=0.5,  class_indexes=1, name="fscore_Kidney")
        fscore_tumor = sm.metrics.FScore(   threshold=0.5,  class_indexes=2, name="fscore_Tumor")
        fscore_cyst = sm.metrics.FScore(    threshold=0.5,  class_indexes=3, name="fscore_Cyst")
    

        metrics = [iou_default, iou_bg, iou_kidney, iou_tumor,iou_cyst,
                    fscore_default, fscore_bg, fscore_kidney,fscore_tumor,fscore_cyst]

        # metrics = [iou_default, iou_bg, iou_kidney, iou_tumor,
        #             fscore_default, fscore_bg, fscore_kidney,fscore_tumor]
        

        # model.compile(optim, total_loss, metrics)
        model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

        pathOutputModel = backboneGeneralPath +"/"+ APPROACH_NAME + "_" + str(RUN) +"/Model/" 
        filename = 'best_model/'+ APPROACH_NAME + "_" + str(RUN)+'/weights.h5'               
        callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ModelCheckpoint(filepath = pathOutputModel+filename, save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.EarlyStopping(patience=8),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',   # Monitora a perda de validação
                factor=0.5,           # Reduz a taxa de aprendizado pela metade
                patience=3,           # Aguarda 3 épocas sem melhora antes de reduzir
                min_lr=1e-7,          # Define o menor valor possível para o LR
                verbose=1             # Exibe logs sobre a redução
            ),
        ]
    
        
        history = model.fit(
            train_dataloader, 
            steps_per_epoch=len(train_dataloader), 
            epochs=EPOCHS, 
            verbose=2,
            callbacks=callbacks, 
            validation_data=valid_dataloader, 
            validation_steps=len(valid_dataloader),
        )

        #model.load_weights(pathOutputModel+filename)  
        # from keras.utils.vis_utils import plot_model
        # model.summary()
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        


        # ################################################################################
        # # Evaluation Train
        # ################################################################################
        pathOutputEvaluationTrain = backboneGeneralPath +"/"+ APPROACH_NAME + "_" + str(RUN) +"/EvaluationTrain/"+ APPROACH_NAME + "_" + str(RUN)+"EvaluateTrain" 
        # Plot training & validation iou_score values
        
        try:
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
            plt.savefig(pathOutputEvaluationTrain+'.png')
            # plt.show()
        except:
            print("Imagem morreu")
    
        
        print("################################################################################\nModel Evaluation with Test Dataset\n################################################################################")
        ################################################################################
        # Model Evaluation with Test Dataset
        ################################################################################
        """# Evaluation on Test Data"""
        pathOutputEvaluationTest = backboneGeneralPath +"/"+ APPROACH_NAME + "_" + str(RUN) +"/EvaluationTest/"+ APPROACH_NAME + "_" + str(RUN) +"EvaluateTest" 
        pathOutputScoreTest = backboneGeneralPath +"/"+ APPROACH_NAME + "_" + str(RUN) +"/Score/"+ APPROACH_NAME + "_" + str(RUN) 
        
        # model.load_weights("/home/viplab/Codes/CentralCode/Results/Kits21/Approaches/DeepLabv3+PPM/DeepLabv3_plus_PPM/DeepLabv3_plus_PPM/Model/best_modelDeepLabv3_plus_PPM.h5") 
        model.load_weights(pathOutputModel+filename)
        scores = model.evaluate_generator(test_dataloader)

        fileScore = open(pathOutputScoreTest + ".txt", "w")
        fileScore.write("#####################\n")
        fileScore.write("Result Segmentation "+CLASSES[0]+" Apply "+ APPROACH_NAME + "_" + str(RUN) +"\n")
        fileScore.write("#####################\n")
        

        # print("Loss: {:.5}".format(scores[0]))
        fileScore.write("Loss: {:.5}".format(scores[0])+"\n")
        for metric, value in zip(metrics, scores[1:]):
            # print("mean {}: {:.5}".format(metric.__name__, value))
            fileScore.write("mean {}: {:.5}".format(metric.__name__, value))
            fileScore.write("\n")
      

        """## Visual Examples on Test Data"""
        from sklearn.metrics import f1_score,accuracy_score,jaccard_score,precision_score,recall_score
        import statistics
        import warnings
        
        imagesList = []
        gtMasksList = []
        prMasksList = []
        scoresListF1 = []
        scoresListACC = []
        scoresListJaccard =[]
        scoresListPrecision=[]
        scoresListRecall=[]


        for samples in (range(len(test_dataset))):
    


            image, gt_mask = test_dataset[samples]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image)
    
            gt_mask = np.argmax(gt_mask, axis = 2)
            pr_mask = np.argmax(pr_mask[0], axis = 2)

            #score = classification_report(gt_mask.flatten(), pr_mask.flatten(), output_dict=True,  zero_division=1)
    
            score_f1 = f1_score(gt_mask.flatten(), pr_mask.flatten(), average='macro', zero_division=1)
            score_acc = accuracy_score(gt_mask.flatten(), pr_mask.flatten())
            score_jacc = jaccard_score(gt_mask.flatten(), pr_mask.flatten(), average='macro', zero_division=1)
            score_prec = precision_score(gt_mask.flatten(), pr_mask.flatten(), average='macro', zero_division=1)
            score_recall = recall_score(gt_mask.flatten(), pr_mask.flatten(), average='macro', zero_division=1)
            
            imagesList.append(image)
            gtMasksList.append(gt_mask)
            prMasksList.append(pr_mask)
            scoresListF1.append(score_f1)
            scoresListACC.append(score_acc)
            scoresListJaccard.append(score_jacc)
            scoresListPrecision.append(score_prec)
            scoresListRecall.append(score_recall)

#        statistics.mean(scoresList)
        fileScore.write("#####################\n")
        fileScore.write("F1-Score: "+str(statistics.mean(scoresListF1)))
        print("F1-Score: "+str(statistics.mean(scoresListF1)))
        fileScore.write("\n")

        fileScore.write("ACC: "+str(statistics.mean(scoresListACC)))
        print("ACC: "+str(statistics.mean(scoresListACC)))
        fileScore.write("\n")
        fileScore.write("Jaccard: "+str(statistics.mean(scoresListJaccard)))
        print("Jaccard: "+str(statistics.mean(scoresListJaccard)))
        fileScore.write("\n")
        fileScore.write("Precision: "+str(statistics.mean(scoresListPrecision)))
        print("Precision: "+str(statistics.mean(scoresListPrecision)))
        fileScore.write("\n")
        fileScore.write("Recall: "+str(statistics.mean(scoresListRecall)))
        print("Recall: "+str(statistics.mean(scoresListRecall)))
        fileScore.write("\n")
        fileScore.write("#####################\n")
        fileScore.close()
        ################################################################################
        # Visualization of results on test dataset
        ################################################################################
        try:
            n = 100
            ids = np.random.choice(np.arange(len(test_dataset)), size=n)

            for i in ids:
                
                image, gt_mask = test_dataset[i]
                if image is None or gt_mask is None:
                    print(f"Entrada inválida no índice {i}")
                    continue
                image = np.expand_dims(image, axis=0)
                pr_mask = model.predict(image)
                pr_mask = model.predict(image)
                pr_mask = np.argmax(pr_mask, axis=-1)
                
                visualize(
                    image=denormalize(image.squeeze()),
                    pr_mask=pr_mask.squeeze(),
                    path=pathOutputEvaluationTest,
                    count = i
                )
                visualize(
                    image=denormalize(image.squeeze()),
                    gt_maskBackground=gt_mask[..., 3].squeeze(),
                    pr_maskBackground=pr_mask[..., 3].squeeze(),
                    gt_maskKidney=gt_mask[..., 0].squeeze(),
                    pr_maskKidney=pr_mask[..., 0].squeeze(),
                    gt_maskTumor=gt_mask[..., 1].squeeze(),
                    pr_maskTumor=pr_mask[..., 1].squeeze(),
                    gt_maskCyst=gt_mask[..., 2].squeeze(),
                    pr_maskCyst=pr_mask[..., 2].squeeze(),
                    path=pathOutputEvaluationTest,
                    count = i
                )
        except:
            print("Nao gerou imagens")
        # for i in ids:
            
        #     image, gt_mask = test_dataset[11179]
        #     #image = np.expand_dims(image, axis=0)
        #     #pr_mask = model.predict(image).round()
        #     pr_mask = model.predict(image)
        #     pr_mask = np.argmax(pr_mask, axis=-1)


        #     visualize(
        #         image=denormalize(image.squeeze()),
        #         gt_maskBackground=gt_mask[..., 3].squeeze(),
        #         pr_maskBackground=pr_mask[..., 3].squeeze(),
        #         gt_maskKidney=gt_mask[..., 0].squeeze(),
        #         pr_maskKidney=pr_mask[..., 0].squeeze(),
        #         gt_maskTumor=gt_mask[..., 1].squeeze(),
        #         pr_maskTumor=pr_mask[..., 1].squeeze(),
        #         gt_maskCyst=gt_mask[..., 2].squeeze(),
        #         pr_maskCyst=pr_mask[..., 2].squeeze(),
        #         path=pathOutputEvaluationTest,
        #         count = 11179
        #     )