"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script performs binary image classification to predict the probability of malignancy for skin lesion images.
It:
- Loads and preprocesses the training data (removes NA labels, drops index column)
- Trains an AutoGluon MultiModalPredictor on the images
- Makes malignancy probability predictions for the test set, preserving original indices and output format
- Saves the trained model and prediction results to the specified output directory
- Performs validation (holdout 10% of training data) and prints AUROC if possible
- Includes validation checks to ensure output correctness

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Place this script in an environment with access to the data folders and run as main.
"""

# Installation steps (uncomment if running in a fresh environment)
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal

import os
import random
import time
import uuid
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Constants
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/28-basic_prompt/node_0/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/28-basic_prompt"
# end change

# Output file name and extension will match test file
RESULTS_BASENAME = "results"

def get_random_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    # Use uuid for randomness, but also add a plausible timestamp for traceability
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:8]
    folder = f"autogluon_model_{ts}_{rand}"
    return os.path.join(base_dir, folder)

def map_label_to_binary(label):
    """Map string label to binary: malignant=1, non-neoplastic=0."""
    # You may need to adjust this mapping if there are more label types
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label == "non-neoplastic":
        return 0
    else:
        # If there are other label types, treat as NA (drop from training)
        return np.nan

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data Loading and Preprocessing
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Map label to binary (malignant=1, non-neoplastic=0), drop NA labels from train only
    train['label'] = train['label'].apply(map_label_to_binary)
    train = train.dropna(subset=['label']).reset_index(drop=True)
    train['label'] = train['label'].astype(int)

    # Add absolute image path for train and test
    train['image'] = train['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))
    # start change
    # test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))  # original (.jpg)
    test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
    # end change

    # Only keep columns needed for training
    train_data = train[['image', 'label']]
    # For prediction, keep image and original index columns for later merging
    test_data = test[['image', 'image_name']].copy()
    test_indices = test.index.copy()  # Save original indices for validation

    # 2. Validation Split (10% holdout if no validation set is provided)
    # Use stratified split to preserve class balance
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42,
        stratify=train_data['label']
    )

    # 3. Model Training
    model_dir = get_random_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)

    # Use default presets for image classification, binary problem
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )

    # Fit model
    predictor.fit(
        train_data=train_df,
        # No need to specify time_limit unless desired
        # Use default presets for robust performance
        # Use only image column and label
    )

    # Save model (already saved by AutoGluon, but ensure it's in the right place)
    # predictor.save()  # Not needed, fit() saves automatically

    # 4. Prediction on Test Set
    # Predict_proba returns probability for each class; for binary, column 1 is "malignant" (label=1)
    proba = predictor.predict_proba(test_data[['image']])
    # proba is a DataFrame with columns [0, 1] (for label=0 and label=1)
    # We want the probability of malignancy (label=1)
    malignancy_probs = proba[1].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_probs.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare result DataFrame
    # Output format: same as test.csv, but with a single column for probability
    # Column name: 'label' (to match train)
    results_df = test[['image_name']].copy()
    results_df['label'] = malignancy_probs

    # Ensure output order matches original test indices
    results_df.index = test_indices

    # Save results in the same format/extension as test.csv
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + test_ext)
    if test_ext == ".csv":
        results_df.to_csv(results_path, index=False)
    elif test_ext in [".tsv", ".txt"]:
        results_df.to_csv(results_path, sep="\t", index=False)
    else:
        # Default to CSV if unknown
        results_df.to_csv(results_path, index=False)

    # 5. Validation Step (AUROC on holdout set)
    try:
        val_pred_proba = predictor.predict_proba(val_df[['image']])
        val_true = val_df['label'].values
        val_pred = val_pred_proba[1].values  # Probability of label=1 (malignant)
        val_auroc = roc_auc_score(val_true, val_pred)
        print(f"Validation AUROC: {val_auroc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 6. Output Validation Checks
    # a) Check number of rows matches test set
    assert len(results_df) == len(test), "Number of predictions does not match number of test samples."
    # b) Check indices preserved
    assert all(results_df.index == test_indices), "Test indices are not preserved in the output."
    # c) Check column names
    assert list(results_df.columns) == ['image_name', 'label'], f"Output columns {list(results_df.columns)} do not match required ['image_name', 'label']"
    # d) Check output format
    assert os.path.exists(results_path), f"Results file not found at {results_path}"
    # e) Check probability values are in [0, 1]
    assert np.all((results_df['label'] >= 0) & (results_df['label'] <= 1)), "Predicted probabilities are not in [0, 1] range."

    print(f"Prediction results saved to: {results_path}")
    print(f"Trained model saved to: {model_dir}")

if __name__ == "__main__":
    main()