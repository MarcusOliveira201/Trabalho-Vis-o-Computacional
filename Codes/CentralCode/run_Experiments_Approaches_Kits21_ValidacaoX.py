import json
import random
import keras
from sklearn.model_selection import KFold
import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
sys.path.insert(1, "C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/DatasetCodes/")
from kitsDataset2 import createBackboneFolders2,defineTrainTestValidation, getImagesPaths, getImagesPathsNew, getListPatient, visualize, Dataset,setResultsPath, setDatabasePath, Dataloader, get_preprocessing, denormalize
sys.path.insert(1, 'C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/CentralCode/utils')
sys.path.insert(1, 'C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/CentralCode/utils')
from utils import getApproach
import segmentation_models as sm
import statistics
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import KFold
import keras.backend as K

# Parâmetros principais
IMAGE_SIZE = 128
BATCH_SIZE = 8
CLASSES = ['background', 'kidney', 'tumor', 'cyst']
N_CLASSES = len(CLASSES)
LR = 0.0001
EPOCHS = 50
ACTIVATION = 'softmax'
APPROACH_NAME = "PPM_SE"
N_FOLDS = 5
resultsPath = "D:/ResultsCrossV/"
setResultsPath(resultsPath)



def create_kfolds(data_dir, k_folds=5):
    """Divide os dados em K folds para cross-validation."""
    jsonFile = data_dir + 'kits.json'
    listIDCases = getListPatient(jsonFile)  # IDs dos casos
    random.shuffle(listIDCases)  # Embaralhar os IDs para aleatoriedade

    kfold = KFold(n_splits=k_folds, shuffle=False)
    folds = []
    for train_idx, val_idx in kfold.split(listIDCases):
        train_cases = [listIDCases[i] for i in train_idx]
        val_cases = [listIDCases[i] for i in val_idx]
        folds.append((train_cases, val_cases))
    
    return folds

data_dir = "C:/Users/suporte/Desktop/Dataset/kits21_Dataset/kits21/kits21/data/"
folds = create_kfolds(data_dir, k_folds=5)  # 5-Fold Cross Validation

optim = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
# class_weights = [1e-6, 1.0, 1.0, 1.0]
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()

focal_loss = sm.losses.BinaryFocalLoss() if N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
#total_loss = focal_tversky_loss
#focal_loss =  sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

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

backboneGeneralPath = createBackboneFolders2(APPROACH_NAME + "_" + str(1))

pathOutputModel = backboneGeneralPath +"/"+ APPROACH_NAME + "_" + str(1) +"/Model/" 
filename = 'best_model/'+ APPROACH_NAME + "_" + str(1)+'/weights.h5'

callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ModelCheckpoint(filepath = pathOutputModel+filename, save_weights_only=False, save_best_only=False, mode='auto'),
            keras.callbacks.EarlyStopping(patience=5),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',   # Monitora a perda de validação
                factor=0.5,           # Reduz a taxa de aprendizado pela metade
                patience=3,           # Aguarda 3 épocas sem melhora antes de reduzir
                min_lr=1e-7,          # Define o menor valor possível para o LR
                verbose=1             # Exibe logs sobre a redução
            ),
        ]


metrics = [iou_default, iou_bg, iou_kidney, iou_tumor,iou_cyst,
            fscore_default, fscore_bg, fscore_kidney,fscore_tumor,fscore_cyst]

for fold, (train_cases, val_cases) in enumerate(folds):
    print(f"Fold {fold+1}:")
    print(f"Treino: {len(train_cases)} casos, Validação: {len(val_cases)} casos")

    # Obter os diretórios de imagens e máscaras
    train_dirs, val_dirs = [], []
    
    x_train_dir = getImagesPathsNew([os.path.join(data_dir, f"{case}/{case}_Exam/") for case in train_cases])
    y_train_dir = getImagesPathsNew([os.path.join(data_dir, f"{case}/{case}_Segmentation/") for case in train_cases])

    x_valid_dir = getImagesPathsNew([os.path.join(data_dir, f"{case}/{case}_Exam/") for case in val_cases])
    y_valid_dir = getImagesPathsNew([os.path.join(data_dir, f"{case}/{case}_Segmentation/") for case in val_cases])

    dataset_test = Dataset(x_train_dir, y_train_dir, classes=['background', 'kidney', 'tumor', 'cyst'])
    print(f"Tamanho do dataset: {len(dataset_test)}")

    # Testando o carregamento do primeiro exemplo
    image, mask = dataset_test[0]
    print(f"Shape da imagem: {image.shape}")
    print(f"Shape da máscara: {mask.shape}")


    train_dataset = Dataset(x_train_dir, y_train_dir, classes=['background', 'kidney', 'tumor', 'cyst'])
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=['background', 'kidney', 'tumor', 'cyst'])

    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

    
    model = getApproach(APPROACH_NAME, IMAGE_SIZE, N_CLASSES, ACTIVATION)
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    print(f"Treinando no Fold {fold+1}...")
    history = model.fit(
        train_dataloader,
        validation_data=valid_dataloader,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Salvar os resultados do fold
    fold_results_path = f"{resultsPath}Fold_{fold+1}/"
    os.makedirs(fold_results_path, exist_ok=True)
    model.save_weights(f"{fold_results_path}model_weights.h5")
    print(f"Resultados do Fold {fold+1} salvos em {fold_results_path}")
    K.clear_session()

all_metrics = []
for fold in range(N_FOLDS):
    metrics_file = f"{resultsPath}Fold_{fold+1}/metrics.json"
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        all_metrics.append(metrics)

avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
print(f"Métricas médias: {avg_metrics}")
