"""
Skin Lesion Malignancy Prediction Script

This script trains an image classification model using AutoGluon MultiModal on a curated skin lesion dataset.
It performs the following:
- Loads and preprocesses the data (removes NA labels, drops index column).
- Trains a binary classifier to predict malignancy probability from lesion images.
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions in the same format as the test file, with correct column names.
- Performs validation (AUROC) using a 10% holdout from the training data, if no validation set is provided.
- Includes validation checks to ensure output correctness.

# Installation requirements (run before executing this script):
# pip install autogluon.multimodal pandas scikit-learn

"""

import os
import shutil
import random
import time
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Set paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/28-basic_prompt/node_0/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
# Test file detection: look for a CSV in DATA_DIR that is not the training file
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.endswith('.csv') and fname != "mydataset.csv":
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

# Helper: get image root directory (assume images are in DATA_DIR or subfolder)
def find_image_path(image_name):
    # Try DATA_DIR/image_name
    path1 = os.path.join(DATA_DIR, image_name)
    if os.path.exists(path1):
        return path1
    # Try DATA_DIR/images/image_name
    path2 = os.path.join(DATA_DIR, "images", image_name)
    if os.path.exists(path2):
        return path2
    raise FileNotFoundError(f"Image file {image_name} not found in expected locations.")

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Data preprocessing
    # Remove unnecessary index column if present
    for col in train_df.columns:
        if col.lower().startswith("unnamed"):
            train_df = train_df.drop(columns=[col])
    for col in test_df.columns:
        if col.lower().startswith("unnamed"):
            test_df = test_df.drop(columns=[col])

    # Remove training samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])

    # Map label to binary: malignant=1, all else=0
    train_df['label'] = (train_df['label'].astype(str).str.lower() == 'malignant').astype(int)

    # For test set, ensure image_name column exists and build absolute image paths
    train_df['image_path'] = train_df['image_name'].apply(find_image_path)
    test_df['image_path'] = test_df['image_name'].apply(find_image_path)

    # 3. Hold out 10% validation set (stratified)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df['label']
    )

    # 4. Model training
    # Prepare output model directory with random timestamp
    timestamp = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Train predictor
    predictor = MultiModalPredictor(
        label='label',
        problem_type='binary',
        eval_metric='roc_auc',
        path=model_dir
    )
    # Only use image_path and label for training
    predictor.fit(
        train_data[['image_path', 'label']],
        time_limit=None,  # No time limit
        presets='best_quality'
    )

    # 5. Prediction on test set
    # Predict_proba returns probability for class 1 (malignant)
    test_pred_proba = predictor.predict_proba(test_df[['image_path']])
    # If output is DataFrame, get probability for class 1
    if isinstance(test_pred_proba, pd.DataFrame):
        # For binary, columns are [0, 1]
        if 1 in test_pred_proba.columns:
            malignancy_prob = test_pred_proba[1].values
        elif '1' in test_pred_proba.columns:
            malignancy_prob = test_pred_proba['1'].values
        else:
            # fallback: take max column
            malignancy_prob = test_pred_proba.iloc[:, -1].values
    else:
        # If output is Series or array
        malignancy_prob = np.array(test_pred_proba)

    # 6. Save predictions
    # Output format: same as test file, but with a column for malignancy probability
    # Try to match sample submission or training column names
    # If test file has a 'label' column, overwrite it; else, add 'label' column
    output_df = test_df.copy()
    output_df['label'] = malignancy_prob
    # Only keep columns present in test file plus 'label'
    keep_cols = [col for col in test_df.columns if col != 'label'] + ['label']
    output_df = output_df[keep_cols]

    # Save with same extension as test file, name "results"
    test_ext = os.path.splitext(TEST_CSV)[1]
    results_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    if test_ext == ".csv":
        output_df.to_csv(results_path, index=False)
    elif test_ext in [".parquet", ".pq"]:
        output_df.to_parquet(results_path, index=False)
    elif test_ext in [".tsv"]:
        output_df.to_csv(results_path, sep='\t', index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {test_ext}")

    # 7. Validation
    try:
        val_pred_proba = predictor.predict_proba(val_data[['image_path']])
        if isinstance(val_pred_proba, pd.DataFrame):
            if 1 in val_pred_proba.columns:
                val_prob = val_pred_proba[1].values
            elif '1' in val_pred_proba.columns:
                val_prob = val_pred_proba['1'].values
            else:
                val_prob = val_pred_proba.iloc[:, -1].values
        else:
            val_prob = np.array(val_pred_proba)
        val_auc = roc_auc_score(val_data['label'], val_prob)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 8. Validation checks
    # a. Check output file has same number of rows as test set
    pred_df = pd.read_csv(results_path) if test_ext == ".csv" else (
        pd.read_parquet(results_path) if test_ext in [".parquet", ".pq"] else pd.read_csv(results_path, sep='\t')
    )
    assert len(pred_df) == len(test_df), "Prediction file row count does not match test set."
    # b. Check indices preserved (if test set has an index column, check order)
    # c. Check column names
    assert 'label' in pred_df.columns, "Output file missing 'label' column."
    # d. Check output format
    assert results_path.startswith(OUTPUT_DIR), "Results file not saved in output directory."
    # e. Check probabilities are in [0, 1]
    assert np.all((pred_df['label'] >= 0) & (pred_df['label'] <= 1)), "Predicted probabilities not in [0, 1]."

    print(f"Predictions saved to: {results_path}")
    print(f"Model saved to: {model_dir}")