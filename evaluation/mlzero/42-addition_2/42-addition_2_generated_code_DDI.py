"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy (malignant vs. non-neoplastic)
for skin lesion images using AutoGluon MultiModal. It:
- Loads and preprocesses the data (removes NA labels, drops index columns)
- Trains a binary classifier on the training set (with 10% held out for validation)
- Saves the trained model to a timestamped folder in the output directory
- Predicts malignancy probability (float in [0,1]) for each test image, preserving original indices and format
- Saves results to the output directory, matching the test file format and column names
- Performs validation checks to ensure output correctness
- Prints AUROC on the held-out validation set (if possible)

# Installation (if needed, run in bash before running this script):
# pip install autogluon.multimodal

Author: AutoML Agent
"""

import os
import time
import random
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_11/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/42-addition_2"
# end change
RESULTS_BASENAME = "results"
MODEL_TIMESTAMP = str(int(time.time())) + f"-{random.randint(1000,9999)}"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, f"automm_model_{MODEL_TIMESTAMP}")

if __name__ == "__main__":
    # 1. Load Data
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # 2. Data Preprocessing
    # Drop index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples with missing labels (but NOT from test)
    train = train.dropna(subset=['label'])

    # Only keep required columns for modeling
    train = train[['image_name', 'label']].copy()
    test = test[['image_name']].copy()

    # Map image_name to absolute path for AutoGluon
    def image_path_fn(x):
        return os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.jpg"))
    train['image'] = train['image_name'].apply(image_path_fn)
    # start change
    # test['image'] = test['image_name'].apply(image_path_fn)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
    )
    # end change

    # Encode label: malignant=1, non-neoplastic=0
    label_map = {'malignant': 1, 'non-neoplastic': 0}
    train['label'] = train['label'].map(label_map)
    train = train[train['label'].isin([0,1])].reset_index(drop=True)

    # 3. Validation Split (10% holdout)
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    idx_train, idx_val = next(splitter.split(train, train['label']))
    train_data = train.iloc[idx_train].reset_index(drop=True)
    val_data = train.iloc[idx_val].reset_index(drop=True)

    # 4. Model Training
    from autogluon.multimodal import MultiModalPredictor

    # Use focal loss to help with class imbalance
    # Compute class weights for focal loss
    counts = train_data['label'].value_counts().sort_index()
    class_weights = [1.0 / float(counts[0]), 1.0 / float(counts[1])]
    class_weights = list(np.array(class_weights) / np.sum(class_weights))
    class_weights = [float(w) for w in class_weights]  # Ensure float, not string

    # Prepare training dataframe for AutoGluon
    ag_train = train_data[['image', 'label']].copy()
    ag_val = val_data[['image', 'label']].copy()

    # Create model save directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=MODEL_SAVE_PATH
    )

    predictor.fit(
        train_data=ag_train,
        tuning_data=ag_val,
        hyperparameters={
            "env.num_gpus": 1,  # Use single GPU
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 15,
            # Model architecture improvements (e.g., use Swin Transformer backbone)
            "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
            "optim.lr": 1e-4,
            "optim.weight_decay": 1e-3,
            "optim.patience": 8,
            "env.precision": "16-mixed",
            "env.num_workers": 4,
        },
        seed=42,
    )

    # 5. Prediction on Test Set
    ag_test = test[['image']].copy()
    y_pred_proba = predictor.predict_proba(ag_test)
    # y_pred_proba is a DataFrame with columns [0, 1] (for each class)
    # We want the probability of class 1 (malignant)
    if 1 in y_pred_proba.columns:
        malignancy_prob = y_pred_proba[1].values
    elif '1' in y_pred_proba.columns:
        malignancy_prob = y_pred_proba['1'].values
    else:
        malignancy_prob = y_pred_proba.iloc[:, 1].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_prob.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # 6. Prepare Output
    results = test.copy()
    results['label'] = malignancy_prob

    # Save results in the same format/extension as test.csv
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + ".csv")
    results.to_csv(results_path, index=False)

    # 7. Validation Checks
    assert len(results) == len(test), f"Number of predictions ({len(results)}) does not match test set ({len(test)})"
    assert all(results['image_name'].values == test['image_name'].values), "Test set indices/order not preserved"
    required_cols = list(test.columns) + ['label']
    assert list(results.columns) == required_cols, f"Output columns {list(results.columns)} do not match required {required_cols}"
    assert results_path.endswith('.csv'), "Output file is not CSV"
    assert np.all((results['label'] >= 0) & (results['label'] <= 1)), "Predicted probabilities not in [0,1]"

    print(f"Prediction results saved to: {results_path}")

    # 8. Validation Metric on Holdout Set
    try:
        from sklearn.metrics import roc_auc_score
        val_pred_proba = predictor.predict_proba(ag_val)
        if 1 in val_pred_proba.columns:
            val_prob = val_pred_proba[1].values
        elif '1' in val_pred_proba.columns:
            val_prob = val_pred_proba['1'].values
        else:
            val_prob = val_pred_proba.iloc[:, 1].values
        val_auc = roc_auc_score(ag_val['label'], val_prob)
        print(f"Validation AUROC on holdout set: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation step failed: {e}")

    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # 9. Provide function for new images (as required in description)
    def predict_malignancy_probabilities(image_folder_path):
        """
        Given a folder path of new images, returns a DataFrame with image_name and malignancy probability (0-1).
        """
        # List all jpg files in the folder
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith('.jpg')]
        df = pd.DataFrame({'image_name': [os.path.splitext(f)[0] for f in image_files]})
        df['image'] = df['image_name'].apply(lambda x: os.path.abspath(os.path.join(image_folder_path, f"{x}.jpg")))
        pred_proba = predictor.predict_proba(df[['image']])
        if 1 in pred_proba.columns:
            prob = pred_proba[1].values
        elif '1' in pred_proba.columns:
            prob = pred_proba['1'].values
        else:
            prob = pred_proba.iloc[:, 1].values
        df['malignancy_probability'] = prob
        return df[['image_name', 'malignancy_probability']]

    print("Script completed successfully.")