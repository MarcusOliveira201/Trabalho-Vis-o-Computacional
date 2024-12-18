import optuna
import sys
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(1, 'C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/CentralCode/utils')
from utils import getApproach

sys.path.insert(1, "C:/Users/suporte/Desktop/Mestrado/caio_falcao_doc/Codes/DatasetCodes/")
from kitsDataset import createBackboneFolders2, defineTrainTestValidation, Dataset, Dataloader, setResultsPath, setDatabasePath

import os

# Define paths
setDatabasePath("C:/Users/suporte/Desktop/Dataset/kits21_Dataset/kits21/kits21/data/")
setResultsPath("D:/Optuna/")

# Fixed parameters
CLASSES = ['background', 'kidney', 'tumor', 'cyst']
IMAGE_SIZE = 128
N_CLASSES = len(CLASSES)
EPOCHS = 10

# Save results in a file
RESULTS_FILE = "optuna_results.txt"

# Objective function
def objective(trial):
    try:
        # Hyperparameters to tune
        APPROACH_NAME = trial.suggest_categorical("approach", ["PPM_SE", "PPM_SE_2"])
        BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8])
        LR = trial.suggest_loguniform("lr", 1e-5, 1e-2)

        # Load data
        x_train_dir, y_train_dir, x_test_dir, y_test_dir, x_valid_dir, y_valid_dir = defineTrainTestValidation(70, 20, 10)
        train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)
        valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES)
        train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

        # Select the model
        model = getApproach(APPROACH_NAME, IMAGE_SIZE, N_CLASSES, "softmax")

        # Compile model
        optim = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        # Custom metrics
        iou_default = sm.metrics.IOUScore(threshold=0.5)
        iou_bg = sm.metrics.IOUScore(threshold=0.5, class_indexes=0, name="iou_BG")
        iou_kidney = sm.metrics.IOUScore(threshold=0.5, class_indexes=1, name="iou_Kidney")
        iou_tumor = sm.metrics.IOUScore(threshold=0.5, class_indexes=2, name="iou_Tumor")
        iou_cyst = sm.metrics.IOUScore(threshold=0.5, class_indexes=3, name="iou_Cyst")

        fscore_default = sm.metrics.FScore(threshold=0.5)
        fscore_bg = sm.metrics.FScore(threshold=0.5, class_indexes=0, name="fscore_BG")
        fscore_kidney = sm.metrics.FScore(threshold=0.5, class_indexes=1, name="fscore_Kidney")
        fscore_tumor = sm.metrics.FScore(threshold=0.5, class_indexes=2, name="fscore_Tumor")
        fscore_cyst = sm.metrics.FScore(threshold=0.5, class_indexes=3, name="fscore_Cyst")

        metrics = [iou_default, iou_bg, iou_kidney, iou_tumor, iou_cyst,
                fscore_default, fscore_bg, fscore_kidney, fscore_tumor, fscore_cyst]

        model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

        # Callbacks
        callbacks = [
            EarlyStopping(patience=3, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)
        ]

        # Train the model
        history = model.fit(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=EPOCHS,
            validation_data=valid_dataloader,
            callbacks=callbacks,
            verbose=1
        )

        # Get the validation loss and metrics
        val_loss = history.history["val_loss"][-1]
        val_iou = history.history.get("val_iou_score", [-1])[-1]
        val_fscore = history.history.get("val_f1-score", [-1])[-1]

        # Print current trial details
        print(f"\n--- Trial {trial.number} ---")
        print(f"Model: {APPROACH_NAME}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Learning Rate: {LR:.6f}")
        print(f"Validation Loss: {val_loss:.5f}")
        print(f"Validation IOU: {val_iou:.5f}")
        print(f"Validation FScore: {val_fscore:.5f}")
        print("---------------------------\n")

        # Save parameters and metrics to file
        with open(RESULTS_FILE, "a") as f:
            f.write("=====================================\n")
            f.write(f"Trial {trial.number}\n")
            f.write(f"Model: {APPROACH_NAME}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Learning Rate: {LR}\n")
            f.write(f"Validation Loss: {val_loss:.5f}\n")
            f.write(f"Validation IOU: {val_iou:.5f}\n")
            f.write(f"Validation FScore: {val_fscore:.5f}\n")
            f.write("=====================================\n\n")

        # Return validation loss (objective to minimize)
    
        return val_loss
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")

# Run Optuna optimization
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)  # Delete the file if it already exists

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, show_progress_bar=True)

# Print all trials after optimization
print("\n=== All Trials ===")
for trial in study.trials:
    print(f"Trial {trial.number}:")
    print(f"  Params: {trial.params}")
    print(f"  Value: {trial.value:.5f}")
print("==================\n")

# Print the best trial
print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value:.5f}")
print(f"  Params: {best_trial.params}")

# Append the best trial to the file
with open(RESULTS_FILE, "a") as f:
    f.write("*************************************\n")
    f.write("Best Trial\n")
    f.write("*************************************\n")
    f.write(f"Value: {best_trial.value:.5f}\n")
    for key, value in best_trial.params.items():
        f.write(f"{key}: {value}\n")
    f.write("*************************************\n")
