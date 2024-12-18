import json
import os, shutil
os.environ["SM_FRAMEWORK"] = "tf.keras"
import random
import os.path
from os import path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pprint
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import segmentation_models as sm
sm.set_framework('tf.keras')
from glob import glob
from collections import defaultdict
import imageio
from PIL import Image
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


#@title ##**Dataset Configuration**

# SIZE IMAGES
HEIGHT_IMG = 128#256
WIDTH_IMG = 128#256
#Percents
randomDataset = True

# Paths
pathDataset = ""
resultsPath = ""


def setDatabasePath(path):
    global pathDataset
    pathDataset = path

def getDatabasePath():
    return pathDataset

def setResultsPath(path):
    global resultsPath
    resultsPath = path

def setImgSize(image_size):
    global HEIGHT_IMG
    global WIDTH_IMG
    HEIGHT_IMG = image_size
    WIDTH_IMG = image_size

def createFolder(pathInput):
     if path.exists(pathInput) == False:
        os.mkdir(pathInput)

"""Create Folders Backbone Output"""
def createBackboneFolders(BACKBONE_TYPE,BACKBONE_LIST):
    backboneGeneralPath = resultsPath + BACKBONE_TYPE
    if path.exists(backboneGeneralPath) == False:
        os.mkdir(backboneGeneralPath)
    print(BACKBONE_LIST)
    for item in BACKBONE_LIST:
        if path.exists(backboneGeneralPath + "/"+ item + "/") == False:
            os.mkdir(backboneGeneralPath + "/"+ item + "/")
            os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/Model/")
            os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/Score/")
            os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/EvaluationTrain/")
            os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/EvaluationTest/")
            os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/Sample/")
            #os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/CSVLogger/")
            #os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/Log/")
    return backboneGeneralPath
    
def createBackboneFolders2(BACKBONE_TYPE):
    backboneGeneralPath = resultsPath + BACKBONE_TYPE
    if path.exists(backboneGeneralPath) == False:
        os.mkdir(backboneGeneralPath)
        os.mkdir(backboneGeneralPath + "/"+"/Model/")
        os.mkdir(backboneGeneralPath + "/"+"/Score/")
        os.mkdir(backboneGeneralPath + "/"+"/EvaluationTrain/")
        os.mkdir(backboneGeneralPath + "/"+"/EvaluationTest/")
        os.mkdir(backboneGeneralPath + "/"+"/Samples/")
        #os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/CSVLogger/")
        #os.mkdir(backboneGeneralPath + "/"+ item + "/"+"/Log/")
    return backboneGeneralPath


"""### **Return Size of Dataset**"""
def getSizeDataset(dataset):
    ids = []
    for i in dataset:
        ids = ids + os.listdir(i)
    return len(ids)

"""### **Dataset Train Test and Validation Ratio**"""
def getListPatient(jsonFile):
        listPatient = []
        with open(jsonFile) as json_file:
            data = json.load(json_file)
            for p in data:
                listPatient.append(p['case_id'])
        return listPatient

def getImagesPaths(caseDir):
    imagePathDir = []

    for j in os.listdir(caseDir):
      imagePathDir.append(os.path.join(caseDir, j))  
    return imagePathDir

def getImagesPathsNew(dirs):
    """Recebe uma lista de diretórios e retorna os caminhos completos das imagens."""
    imagePathDir = []
    for caseDir in dirs:
        if os.path.exists(caseDir):  # Verifica se o diretório existe
            for file in os.listdir(caseDir):
                imagePathDir.append(os.path.join(caseDir, file))
        else:
            print(f"Aviso: Diretório {caseDir} não encontrado.")
    return imagePathDir


def createDirectory(path):
    if not os.path.exists(path):
      os.makedirs(path)

def defineTrainTestValidation(trainPercents,testPercents,validationPercents):

    print("###########",pathDataset)

    if "21" in pathDataset:
        jsonFile = pathDataset +'kits.json'
        listIDCases = getListPatient(jsonFile)
        sizeDataset = len(listIDCases)
    else:
        listIDCases = os.listdir(pathDataset)
        listIDCases.sort()
        sizeDataset = len(listIDCases)
    typeImages = ['Exam','Segmentation'] 
    ########################################### Train #################################################
    listCaseTrainExam = []
    listCaseTrainMasks = []

    sizeTrain = int((trainPercents/100) * sizeDataset)


    subListIDCasesTrain = []
    print("#####################################")
    print("Total List Case: ",len(listIDCases))
    print("#####################################\n")
    print("Size Train: ",sizeTrain)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeTrain):
            case = random.choice(listIDCases)
            #print("Case Random",case)
            subListIDCasesTrain.append(case)
            listIDCases.remove(case)
    ### Without Random Pacients
    else:
        for i in range(sizeTrain):
            case = listIDCases[0]
            subListIDCasesTrain.append(case)
            #print(case)
            listIDCases.remove(case)

    #print(len(subListIDCasesTrain))
    for case in subListIDCasesTrain:
        #print("@@@@@@@@@@@@@@@@@@@@ Dataset:",case)
        #Exam Image
        pathImageExam   = pathDataset +  case + '/' + case + '_' + typeImages[0] + '/'
        #pathImageExam   = pathDataset +  case + '/'  + typeImages[0] + '/'
        listCaseTrainExam.append(pathImageExam)

        #Masks Kidney Add Tumor 
        pathImageMasks   = pathDataset +  case + '/' + case + '_' + typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/'  + typeImages[1] + '/'
        listCaseTrainMasks.append(pathImageMasks)

    ########################################### Test #################################################
    listCaseTestExam = []
    listCaseTestMasks = []
    test_cases = []
    sizeTest = int((testPercents/100) * sizeDataset)

    subListIDCasesTest = []
    print("Current Size List Case: ",len(listIDCases))
    print("Size Test: ",sizeTest)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeTest):
            case = random.choice(listIDCases)
            #print("Case Random",case)
            subListIDCasesTest.append(case)
            listIDCases.remove(case)
    ### Without Random Pacients
    else:
        for i in range(sizeTest):
            case = listIDCases[0]
            subListIDCasesTest.append(case)
            listIDCases.remove(case)

    #print(len(listIDCases))
    #print(len(subListIDCasesTest))
    for case in subListIDCasesTest:
        test_cases.append(case)
        #Exam Image
        pathImageExam   = pathDataset +  case + '/' + case + '_' + typeImages[0] + '/'
        #pathImageExam   = pathDataset +  case + '/' + typeImages[0] + '/'
        listCaseTestExam.append(pathImageExam)
        
        #Kidney Mask And Tumor Mask
        pathImageMasks   = pathDataset +  case + '/' + case + '_' + typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/' + typeImages[1] + '/'
        listCaseTestMasks.append(pathImageMasks)

    ########################################### Validation ################################################# 
    listCaseValidExam = []
    listCaseValidMasks = []

    sizeValidation = int((validationPercents/100) * sizeDataset)


    subListIDCasesValidation = []
    print("Current Size List Case: ",len(listIDCases))
    print("Size Validation: ",sizeValidation)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeValidation):
            case = random.choice(listIDCases)
            subListIDCasesValidation.append(case)
            listIDCases.remove(case)
    ### Without Random Pacients
    else:
        for i in range(sizeValidation):
            case = listIDCases[0]
            subListIDCasesValidation.append(case)
            listIDCases.remove(case)

    print(len(listIDCases))
    print(len(subListIDCasesValidation))
    for case in subListIDCasesValidation:
        
        #Kidney Image
        pathImageKidney   = pathDataset +  case + '/' + case + '_' + typeImages[0] + '/'
        #pathImageKidney   = pathDataset +  case + '/'  + typeImages[0] + '/'
        listCaseValidExam.append(pathImageKidney)
        
        #Kidney Mask Add Tumor Mask
        pathImageMasks   = pathDataset +  case + '/' + case + '_' + typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/'  + typeImages[1] + '/'
        listCaseValidMasks.append(pathImageMasks)
    listCaseTrainExam.sort()
    listCaseTrainMasks.sort()
    listCaseTestExam.sort()
    listCaseTestMasks.sort()
    listCaseValidExam.sort()
    listCaseValidMasks.sort()

    x_train_dir = listCaseTrainExam
    y_train_dir = listCaseTrainMasks
    x_test_dir = listCaseTestExam
    y_test_dir = listCaseTestMasks
    x_valid_dir = listCaseValidExam
    y_valid_dir = listCaseValidMasks
    return x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir#,test_cases

################################################################################
# New DataLoader
################################################################################
# helper function for data visualization
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# def visualize( path, count,**images):
#     """PLot images in one row."""
#     n = len(images)
#     plt.figure(figsize=(30, 5))
#     for i, (name, image) in enumerate(images.items()):
#         plt.subplot(1, n, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(' '.join(name.split('_')).title())
#         #plt.imshow(image, cmap='gray' )
#         plt.imshow(image)
#         plt.savefig(path + str(count) +'.png')
    # plt.show()
    #plt.savefig(path + str(count) +'.png')]
def visualize(path, count, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(30, 5))
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        # plt.imshow(image, cmap='gray')  # ou remova cmap conforme necessário
        plt.imshow(image)
    plt.savefig(f"{path}{count}.png")  # Salva a figura completa, fora do loop
    plt.close()  # Fecha a figura para liberar memória
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    CLASSES = ['background', 'kidney', 'tumor', 'cyst']
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        # Como os caminhos já são completos, basta usá-los diretamente
        self.images_fps = images_dir  # x_train_dir já contém os caminhos completos das imagens
        self.masks_fps = masks_dir    # y_train_dir já contém os caminhos completos das máscaras

        # IDs são os nomes dos arquivos, sem os diretórios
        self.ids = [os.path.basename(fp) for fp in self.images_fps]

        # Depuração
        print(f"Imagens carregadas: {len(self.images_fps)}")
        print(f"Máscaras carregadas: {len(self.masks_fps)}")
        print(f"IDs: {self.ids[:5]}")

        # Checagem básica
        if not self.images_fps or not self.masks_fps:
            raise ValueError("Dataset está vazio. Verifique os caminhos e os dados.")

        assert len(self.images_fps) == len(self.masks_fps), \
            "Número de imagens e máscaras não corresponde."

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing 
        
    def __getitem__(self, i):
        
        # read data
        #print(i,len(self.images_fps))

        # print(i)
        image = cv2.imread(self.images_fps[i])
        # print(image.shape)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (HEIGHT_IMG, WIDTH_IMG)) 
 

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (HEIGHT_IMG, WIDTH_IMG)) 
 
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # #add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))  # Inicializar com os índices do dataset

        if len(self.indexes) == 0:
            raise ValueError("O Dataloader foi inicializado com um dataset vazio.")

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])  # Acessar dados pelo índice
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)




################################################################################
# Augmentations
################################################################################
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

#####################################################
# Novas funções de Carregamento do Dataset
#####################################################

#CLASSES = ['background','kidney','tumor','cyst']
def createFoldersOutput(APPROACHE_TYPE):
    rootPath = resultsPath + APPROACHE_TYPE
    if path.exists(rootPath) == False:
        os.mkdir(rootPath)
        os.mkdir(rootPath + "/"+"/Model/")
        os.mkdir(rootPath + "/"+"/Score/")
        os.mkdir(rootPath + "/"+"/EvaluationTrain/")
        os.mkdir(rootPath + "/"+"/EvaluationTest/")
        os.mkdir(rootPath + "/"+"/Samples/")
        #os.mkdir(rootPath + "/"+ item + "/"+"/CSVLogger/")
        #os.mkdir(rootPath + "/"+ item + "/"+"/Log/")
    return rootPath

def splitTrainTestValidate(pathDataset,train_rate,test_rate,valid_rate,shuffle=False):

    print("###########",pathDataset)
    
    print(pathDataset)
    jsonFile = pathDataset +'kits.json'
    print(jsonFile)
    listIDCases = getListPatient(jsonFile)
    typeImages = ['Exam','Segmentation'] 

    listCaseExams = []
    listCaseMasks = []


    for case in listIDCases:
        #Imgs 
        pathImageExam   = pathDataset +  case + '/' + case + '_' + typeImages[0] + '/'
        listCaseExams.append(pathImageExam)
        #Masks 
        pathImageMasks   = pathDataset +  case + '/' + case + '_' + typeImages[1] + '/'
        listCaseMasks.append(pathImageMasks)

    
    # listCaseExams.sort()
    # listCaseMasks.sort()
    

    x_dir = listCaseExams
    y_dir = listCaseMasks
   
    #Separando train teste e validação por paciente
    df = pd.DataFrame(data={"filename": x_dir, 'mask' : y_dir})
    train_split_pacients, test_split_pacients = train_test_split(df,test_size = test_rate,shuffle=False)
    train_split_pacients, val_split_pacients = train_test_split(train_split_pacients,test_size = valid_rate,shuffle=False)

    
    print(train_split_pacients.values.shape)
    print(val_split_pacients.values.shape)
    print(test_split_pacients.values.shape)


    ######### Train ############
    list_train_x = []
    list_train_y = []
    
    for i in range(0,len(train_split_pacients)):
        img = train_split_pacients['filename'].iloc[i]
        mask = train_split_pacients['mask'].iloc[i]          
        x = glob(img+'/*.png')
        x.sort()
        for file  in x:
            list_train_x.append(file)
        y = glob(mask+'/*.png')
        y.sort()
        for file  in y:
            list_train_y.append(file)

    # for i in range(1200,1300):
    #     visualize_data_temp("train",list_train_x[i],list_train_y[i],i)
    
    ######### Test ############
    list_test_x = []
    list_test_y = []
    
    for i in range(0,len(test_split_pacients)):
        img = test_split_pacients['filename'].iloc[i]
        mask = test_split_pacients['mask'].iloc[i]          
        x = glob(img+'/*.png')
        for file  in x:
            list_test_x.append(file)
        y = glob(mask+'/*.png')
        for file  in y:
            list_test_y.append(file)

    #visualize_data_temp("teste",list_test_x[0],list_test_y[0],0)
    ######### Validate ############
    list_val_x = []
    list_val_y = []
    
    for i in range(0,len(val_split_pacients)):
        img = val_split_pacients['filename'].iloc[i]
        mask = val_split_pacients['mask'].iloc[i]          
        x = glob(img+'/*.png')
        for file  in x:
            list_val_x.append(file)
        y = glob(mask+'/*.png')
        for file  in y:
            list_val_y.append(file)

    #visualize_data_temp("valid",list_val_x[0],list_val_y[0],0)

    df_valid = pd.DataFrame(data={"filename": list_val_x, 'mask' : list_val_y})  
    df_test = pd.DataFrame(data={"filename": list_test_x, 'mask' : list_test_y})  
    df_train = pd.DataFrame(data={"filename": list_train_x, 'mask' : list_train_y})  
    return df_train,df_test,df_valid



def visualize_data(train_files,outputImage,image_size,image_index):
    #image_size = 32
    #image_index = 500

    # create figure 
    fig = plt.figure(figsize=(20, 7)) 
    
    # reading images
    img_path = train_files['filename'].iloc[image_index]
    msk_path = train_files['mask'].iloc[image_index] 
    
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((image_size,image_size))
    img = np.reshape(img,(image_size,image_size,3))
    img = img/256

    mask = Image.open(msk_path).convert("L")
    mask = mask.resize((image_size, image_size))
    mask = np.array(mask)
    
    
    mask[mask > 1] = 0
    mask[mask != 0] = 1
    # msk = msk/255.0
    mask = np.reshape(mask,(image_size,image_size,1))
    

    
    #Lets plot some samples    
    rows = 1
    columns = 2
   
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(img)
    # plt.title("Image") 

    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask)
    plt.title("Mask") 

    fig.add_subplot(rows, columns, 1)
    df_cm = pd.DataFrame(mask[:,:,0])
    sn.heatmap(df_cm, annot=True)
    # plt.imshow(mask)
    plt.title("Pixel Visualize Mask") 
    
    
    
    #plt.show()
    
    plt.savefig(outputImage+'/sampleDataset_'+str(image_index)+'.png')

def visualize_data_temp(data_files,output):
    x = next(data_files)
    # x= data_files.next()
    for i in range(0,4):
        image = x[i]
        #print(image)
        plt.imshow(image)
        # plt.show()
        # plt.savefig(output+'/sampleDataset_'+str(i)+'.png')

def Data_Generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    #train_gen = zip(image_generator, mask_generator)
    train_gen = (pair for pair in zip(image_generator, mask_generator))
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    #mask = mask / 255
    mask[mask > 1] = 0
    mask[mask != 0] = 1
    
    return (img, mask)


