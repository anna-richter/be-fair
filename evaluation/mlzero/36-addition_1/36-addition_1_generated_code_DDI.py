"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy (malignant vs. benign)
for skin lesion images, using the AutoGluon MultiModal framework. It performs the following steps:

1. Data preprocessing:
   - Loads train and test CSVs, removes samples with missing labels from train only.
   - Removes unnecessary index columns.
   - Maps image names to absolute image paths.
   - Converts string labels to binary (malignant=1, non-neoplastic=0).
   - Ensures test set has all required columns (e.g., skin_tone).
2. Model training:
   - Trains a MultiModalPredictor for binary classification, using focal loss with class weights for fairness.
   - Saves the trained model to a timestamped folder in the specified output directory.
   - Holds out 10% of the training data for validation if no separate validation set is provided.
3. Prediction:
   - Predicts malignancy probability (float in [0,1]) for each test image, preserving original indices.
   - Saves results in the same format and extension as the test file, with correct column names.
4. Validation:
   - Computes and prints AUROC on the held-out validation set.
5. Validation checks:
   - Ensures output file matches test indices, column names, format, and row count.
   - Checks that all probabilities are in [0,1].

Installation requirements:
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Place this script in the working directory and run it. Ensure data is present at the specified paths.

Author: AutoML Agent
"""

# Installation steps (uncomment if running in a fresh environment)
# !pip install --upgrade pip
# !pip install autogluon.multimodal

import os
import time
import uuid
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Paths
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/36-addition_1/node_8/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/36-addition_1"
# end change

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time())
    rand = uuid.uuid4().hex[:8]
    folder = f"autogluon_model_{ts}_{rand}"
    return os.path.join(base_dir, folder)

def map_image_name_to_path(df, image_col, img_dir):
    """Map image_name column to absolute image paths."""
    df = df.copy()
    df[image_col] = df[image_col].apply(lambda x: os.path.abspath(os.path.join(img_dir, f"{x}.jpg")))
    return df

def prepare_train_data(train_csv, img_dir):
    """Load and preprocess training data."""
    df = pd.read_csv(train_csv)
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Drop rows with missing labels (train only)
    df = df.dropna(subset=['label'])
    # Map image_name to absolute path
    df = map_image_name_to_path(df, 'image_name', img_dir)
    # Only keep necessary columns
    keep_cols = ['image_name', 'skin_tone', 'label']
    df = df[[col for col in keep_cols if col in df.columns]]
    # Convert label to binary: malignant=1, non-neoplastic=0
    df['label'] = df['label'].map(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)
    # Fill missing skin_tone with median (for fairness)
    if df['skin_tone'].isnull().any():
        df['skin_tone'] = df['skin_tone'].fillna(df['skin_tone'].median())
    return df

def prepare_test_data(test_csv, img_dir, train_skin_tone_median):
    """Load and preprocess test data."""
    # start change
    # df = pd.read_csv(test_csv)
    df = pd.read_csv(test_csv, dtype={"image_name": str})
    # end change
    orig_index = df.index.copy()
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Map image_name to absolute path
    df = map_image_name_to_path(df, 'image_name', img_dir)
    # Ensure skin_tone column exists (fill with train median if missing)
    if 'skin_tone' not in df.columns:
        df['skin_tone'] = train_skin_tone_median
    else:
        df['skin_tone'] = df['skin_tone'].fillna(train_skin_tone_median)
    return df, orig_index

def compute_class_weights(labels):
    """Compute class weights for focal loss (for fairness)."""
    counts = np.bincount(labels)
    total = counts.sum()
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    weights = np.array(weights) / np.sum(weights)
    return weights.tolist()

def save_results(df, test_csv, output_dir, col_name='label'):
    """Save results in the same format and extension as test_csv, with correct column names."""
    ext = os.path.splitext(test_csv)[1]
    out_path = os.path.join(output_dir, "results" + ext)
    # start change
    # test_df = pd.read_csv(test_csv)
    test_df = pd.read_csv(test_csv, dtype={"image_name": str})
    # end change
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    test_df[col_name] = df[col_name].values
    if ext == ".csv":
        test_df.to_csv(out_path, index=False)
    elif ext in [".tsv", ".txt"]:
        test_df.to_csv(out_path, sep='\t', index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {ext}")
    return out_path

def validate_output(pred_path, test_csv, col_name='label'):
    """Validation checks on the prediction file."""
    # start change
    # test_df = pd.read_csv(test_csv)
    test_df = pd.read_csv(test_csv, dtype={"image_name": str})
    # end change
    pred_df = pd.read_csv(pred_path)
    for col in ['Unnamed: 0']:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
        if col in pred_df.columns:
            pred_df = pred_df.drop(columns=[col])
    assert len(pred_df) == len(test_df), f"Prediction rows {len(pred_df)} != test rows {len(test_df)}"
    assert (pred_df.index == test_df.index).all(), "Prediction indices do not match test indices"
    for col in test_df.columns:
        assert col in pred_df.columns, f"Missing column {col} in predictions"
    assert col_name in pred_df.columns, f"Missing prediction column {col_name}"
    probs = pred_df[col_name].values
    assert np.all((probs >= 0) & (probs <= 1)), "Predicted probabilities not in [0,1]"
    print("Validation checks passed: output file matches test indices, columns, format, and value range.")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data Loading and Preprocessing
    train_df = prepare_train_data(TRAIN_CSV, IMG_DIR)
    train_skin_tone_median = train_df['skin_tone'].median()
    test_df, test_orig_index = prepare_test_data(TEST_CSV, IMG_DIR, train_skin_tone_median)
    # start change
    test_df["image_name"] = test_df["image_name"].apply(
        lambda p: os.path.splitext(p)[0] + ".png"
    )
    # end change

    # 2. Train/Validation Split (if no validation set is provided)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, test_size=0.10, random_state=42, stratify=train_df['label']
    )

    # 3. Compute class weights for fairness (focal loss)
    class_weights = compute_class_weights(train_data['label'].values)

    # 4. Model Training
    model_dir = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )
    # Use focal loss for fairness, include skin_tone as tabular feature
    predictor.fit(
        train_data=train_data,
        time_limit=3600-600,  # leave buffer for inference
        hyperparameters={
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 5,  # Reduce epochs for faster training
            # Optionally, use a robust image backbone:
            # "model.timm_image.checkpoint_name": "mobilenetv3_large_100",
        },
        tuning_data=val_data,  # Use tuning_data instead of val_data
    )

    # 5. Prediction on Test Set
    # Prepare test data for prediction (must match train columns except label)
    test_pred_df = test_df.copy()
    # Predict probabilities for class 1 (malignant)
    proba = predictor.predict_proba(test_pred_df)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            test_pred_df['label'] = proba[1].values
        elif 'malignant' in proba.columns:
            test_pred_df['label'] = proba['malignant'].values
        else:
            test_pred_df['label'] = proba.iloc[:, -1].values
    else:
        test_pred_df['label'] = proba

    # Restore original test indices
    test_pred_df.index = test_orig_index

    # start change
    # _orig_test_names = pd.read_csv(TEST_CSV)["image_name"].astype(str).reset_index(drop=True)
    _orig_test_names = pd.read_csv(TEST_CSV, dtype={"image_name": str})["image_name"].astype(str).reset_index(drop=True)
    _ddi_df = pd.DataFrame({
        "DDI_file": _orig_test_names + ".png",
        "predicted_probability": test_pred_df["label"].astype(float).values,
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # 6. Save Results
    results_path = save_results(test_pred_df[['label']], TEST_CSV, OUTPUT_DIR, col_name='label')

    # 7. Validation Checks
    validate_output(results_path, TEST_CSV, col_name='label')

    # 8. Validation Metric on Held-out Set
    try:
        val_pred_proba = predictor.predict_proba(val_data)
        if isinstance(val_pred_proba, pd.DataFrame):
            if 1 in val_pred_proba.columns:
                val_probs = val_pred_proba[1].values
            elif 'malignant' in val_pred_proba.columns:
                val_probs = val_pred_proba['malignant'].values
            else:
                val_probs = val_pred_proba.iloc[:, -1].values
        else:
            val_probs = val_pred_proba
        val_labels = val_data['label'].values
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation step failed: {e}")

    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")