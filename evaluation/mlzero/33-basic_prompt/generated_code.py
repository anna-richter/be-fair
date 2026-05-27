"""
Skin Lesion Malignancy Prediction Script

This script trains an image classification model using AutoGluon MultiModal to predict the probability of malignancy for skin lesion images.
It performs the following steps:
- Loads and preprocesses the training and test data (removes NA labels from train, drops index columns, constructs absolute image paths).
- Maps all non-malignant labels ("benign", "non-neoplastic") to 0, and "malignant" to 1.
- Trains a binary classifier (malignant vs. benign) using the provided images.
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation (10% holdout) if training labels are available, printing AUROC on the validation set.
- Includes validation checks to ensure output correctness.

Installation requirements:
    # pip install --upgrade pip
    # pip install autogluon.multimodal

Usage:
    Place this script in your working environment and run it as a standalone program.
    Ensure the data folder and output directory exist and are accessible.

Author: AutoML Agent
"""

import os
import random
import warnings
import pandas as pd
import numpy as np

from datetime import datetime
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ==== CONFIGURATION ====
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/33-basic_prompt/node_8/output"
RESULTS_BASENAME = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_timestamp_folder(base_dir):
    """Create a random timestamped folder for model saving."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = random.randint(1000, 9999)
    folder = os.path.join(base_dir, f"autogluon_model_{timestamp}_{rand}")
    os.makedirs(folder, exist_ok=True)
    return folder

def map_label_to_binary(label):
    """Map string label to binary: malignant=1, benign/non-neoplastic=0."""
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label in ["non-neoplastic", "benign"]:
        return 0
    else:
        raise ValueError(f"Unknown label encountered: {label}")

def get_test_file_format(test_csv_path):
    """Determine if test file is CSV or something else."""
    _, ext = os.path.splitext(test_csv_path)
    return ext.lower()

def get_output_file_path(test_csv_path):
    """Return output file path with same extension as test file."""
    ext = get_test_file_format(test_csv_path)
    return os.path.join(OUTPUT_DIR, RESULTS_BASENAME + ext)

def get_column_names_from_file(csv_path):
    """Read only the header row to get column names."""
    return pd.read_csv(csv_path, nrows=0).columns.tolist()

def get_image_abs_path(image_name):
    """Return absolute path to image file given its name (with or without .jpg)."""
    if not image_name.lower().endswith('.jpg'):
        image_name = image_name + ".jpg"
    return os.path.abspath(os.path.join(IMG_DIR, image_name))

def prepare_train_data(train_csv):
    """Load and preprocess training data."""
    df = pd.read_csv(train_csv)
    # Remove index column if present
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    # Remove samples with missing labels
    df = df.dropna(subset=['label'])
    # Map labels to binary
    df['label'] = df['label'].map(map_label_to_binary)
    # Remove any rows where mapping failed (shouldn't happen)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    # Add absolute image path column
    df['image'] = df['image_name'].apply(get_image_abs_path)
    # Remove image_name column (AutoGluon expects 'image')
    df = df.drop(columns=['image_name'])
    return df

def prepare_test_data(test_csv):
    """Load and preprocess test data. Do NOT drop any rows."""
    df = pd.read_csv(test_csv)
    orig_index = df.index.copy()
    # Remove index column if present
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    # Add absolute image path column
    df['image'] = df['image_name'].apply(get_image_abs_path)
    # Remove image_name column (AutoGluon expects 'image')
    df = df.drop(columns=['image_name'])
    # Keep original indices for validation
    df.index = orig_index
    return df

def get_label_column_name(train_csv):
    """Return the label column name as in the training file."""
    cols = get_column_names_from_file(train_csv)
    if 'label' in cols:
        return 'label'
    else:
        raise ValueError("No 'label' column found in training data.")

def get_test_id_column_names(test_csv):
    """Return the identifier columns in the test file (excluding image_name)."""
    cols = get_column_names_from_file(test_csv)
    id_cols = [c for c in cols if c != 'image_name']
    id_cols = [c for c in id_cols if c not in ['Unnamed: 0', 'index']]
    return id_cols

def get_output_column_names(train_csv, test_csv):
    """Determine output column names: test id columns + label column."""
    test_id_cols = get_test_id_column_names(test_csv)
    label_col = get_label_column_name(train_csv)
    return test_id_cols + [label_col]

def save_predictions_with_format(test_csv, preds, output_path, label_col):
    """Save predictions to output_path, matching test file's format and columns."""
    test_df = pd.read_csv(test_csv)
    orig_index = test_df.index.copy()
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in test_df.columns:
            test_df = test_df.drop(columns=[idx_col])
    out_df = test_df.copy()
    id_cols = get_test_id_column_names(test_csv)
    id_cols = [c for c in id_cols if c in out_df.columns]
    output_df = out_df[id_cols].copy() if id_cols else pd.DataFrame(index=out_df.index)
    output_df[label_col] = preds
    assert len(output_df) == len(test_df)
    output_df.index = orig_index
    ext = get_test_file_format(test_csv)
    if ext == ".csv":
        output_df.to_csv(output_path, index=False)
    else:
        raise NotImplementedError(f"Test file format {ext} not supported.")
    return output_df

def validate_output(test_csv, output_path, label_col):
    """Validation checks on the output predictions file."""
    test_df = pd.read_csv(test_csv)
    pred_df = pd.read_csv(output_path)
    assert len(test_df) == len(pred_df), f"Prediction rows ({len(pred_df)}) != test rows ({len(test_df)})"
    test_id_cols = get_test_id_column_names(test_csv)
    test_id_cols = [c for c in test_id_cols if c in pred_df.columns]
    expected_cols = test_id_cols + [label_col]
    assert list(pred_df.columns) == expected_cols, f"Output columns {list(pred_df.columns)} != expected {expected_cols}"
    assert output_path.endswith(".csv"), "Output file must be CSV."
    preds = pred_df[label_col].values
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions not in [0,1] range."
    assert len(pred_df.columns) == len(expected_cols), "Extra columns in output."
    print("Output validation checks passed.")

def compute_validation_score(predictor, val_df, label_col):
    """Compute AUROC on validation set."""
    try:
        y_true = val_df[label_col].values
        proba_df = predictor.predict_proba(val_df)
        if 1 in proba_df.columns:
            y_pred = proba_df[1].values
        elif "1" in proba_df.columns:
            y_pred = proba_df["1"].values
        else:
            y_pred = proba_df.iloc[:, 1].values
        score = roc_auc_score(y_true, y_pred)
        print(f"Validation AUROC: {score:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    # 1. Load and preprocess data
    train_df = prepare_train_data(TRAIN_CSV)
    test_df = prepare_test_data(TEST_CSV)
    label_col = get_label_column_name(TRAIN_CSV)
    output_colnames = get_output_column_names(TRAIN_CSV, TEST_CSV)
    output_path = get_output_file_path(TEST_CSV)

    # 2. Hold out 10% validation if no explicit validation set
    if len(train_df) > 10:
        train_data, val_data = train_test_split(
            train_df, test_size=0.10, random_state=42, stratify=train_df[label_col]
        )
    else:
        train_data = train_df
        val_data = None

    # 3. Train model
    model_save_dir = get_random_timestamp_folder(OUTPUT_DIR)
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_save_dir
    )
    train_cols = ['image', label_col]
    predictor.fit(
        train_data=train_data[train_cols],
        time_limit=3600-300,  # leave buffer for inference
    )

    # 4. Predict on test set (probability of malignancy)
    proba_df = predictor.predict_proba(test_df)
    if 1 in proba_df.columns:
        malignancy_proba = proba_df[1].values
    elif "1" in proba_df.columns:
        malignancy_proba = proba_df["1"].values
    else:
        malignancy_proba = proba_df.iloc[:, 1].values

    # 5. Save predictions in required format
    save_predictions_with_format(TEST_CSV, malignancy_proba, output_path, label_col)

    # 6. Validation checks
    validate_output(TEST_CSV, output_path, label_col)

    # 7. Validation metric on holdout set
    if val_data is not None and len(val_data) > 0:
        print("Computing validation AUROC on holdout set...")
        compute_validation_score(predictor, val_data[['image', label_col]], label_col)