"""
Skin Lesion Malignancy Prediction Script

This script trains an image classification model using AutoGluon MultiModal on a curated dataset of skin lesion images.
It predicts the probability of malignancy for each lesion in the test set, saving results in the required format.
The script ensures fairness by preserving all test samples and indices, and includes validation and output checks.

Installation requirements (run before executing this script):
    pip install autogluon.multimodal==0.9.0 pandas scikit-learn

Data and output locations are hardcoded as per the task specification.
"""

# Installation instructions (uncomment and run if needed)
# !pip install autogluon.multimodal==0.9.0 pandas scikit-learn

import os
import random
import time
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_3/output"
IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Output file name and path
RESULTS_FILENAME = "results"
# We'll determine the extension based on test.csv

def get_label_map():
    """
    Returns a mapping from string labels to binary values.
    'malignant' -> 1, all others (including 'benign', 'non-neoplastic') -> 0
    """
    return {'malignant': 1, 'benign': 0, 'non-neoplastic': 0}

def check_label_coverage(df):
    """
    Checks that all label values are covered by the label mapping.
    Raises ValueError if any unmapped labels are found.
    """
    label_map = get_label_map()
    unique_labels = set(df['label'].dropna().unique())
    missing = unique_labels - set(label_map.keys())
    if missing:
        raise ValueError(f"Unmapped label values found in training data: {missing}")

def prepare_dataframe(df, is_train=True):
    """
    Prepares the dataframe for AutoGluon:
    - Drops NA labels (train only)
    - Removes index column if present
    - Maps label to binary (malignant=1, else=0)
    - Adds absolute image path column
    """
    df = df.copy()
    # Remove index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Drop NA labels for training only
    if is_train:
        df = df.dropna(subset=['label'])
        check_label_coverage(df)
        label_map = get_label_map()
        df['label'] = df['label'].map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
    # Add absolute image path
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
    return df

def get_results_extension(test_csv_path):
    """
    Returns the file extension of the test file (e.g., '.csv').
    """
    _, ext = os.path.splitext(test_csv_path)
    return ext

def get_pred_column_name(train_df):
    """
    Returns the column name for the prediction output.
    Should match the label column in train/test/sample submission.
    """
    # In this dataset, it's 'label'
    return 'label'

def main():
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Preprocess data
    train_df = prepare_dataframe(train_df, is_train=True)
    test_df_orig = test_df.copy()  # For index preservation and validation
    test_df = prepare_dataframe(test_df, is_train=False)

    # 3. Split train/validation
    # Only if there is labeled training data and no separate validation set
    # We'll stratify by label to preserve class balance
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        random_state=SEED,
        stratify=train_df['label']
    )

    # 4. Train model
    # Save model to a random timestamped folder in OUTPUT_DIR
    timestamp = int(time.time()) + random.randint(0, 100000)
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # AutoGluon expects:
    # - image column (path to image)
    # - label column (classification target)
    predictor = MultiModalPredictor(
        label='label',
        problem_type='binary',
        path=model_dir,
        eval_metric='roc_auc',
        seed=SEED,
    )

    # Train
    predictor.fit(
        train_data=train_data,
        time_limit=None,  # No time limit
        presets='best_quality',
        tuning_data=val_data,
        hyperparameters=None,
    )

    # 5. Predict on test set
    # Ensure we predict on the full test set, preserving original indices
    # We'll use predictor.predict_proba to get probability for class 1 (malignant)
    # The output must have the same number of rows and indices as test_df_orig

    # AutoGluon expects the image path column to be present
    test_pred_proba = predictor.predict_proba(test_df, as_pandas=True)
    # test_pred_proba is a DataFrame with columns [0, 1] for class probabilities

    # We want the probability for class 1 (malignant)
    if 1 in test_pred_proba.columns:
        malignancy_proba = test_pred_proba[1]
    elif '1' in test_pred_proba.columns:
        malignancy_proba = test_pred_proba['1']
    else:
        # fallback: take the last column (should be class 1)
        malignancy_proba = test_pred_proba.iloc[:, -1]

    # Prepare results DataFrame
    results_df = test_df_orig.copy()
    pred_col = get_pred_column_name(train_df)
    results_df[pred_col] = malignancy_proba.values

    # Ensure output format matches test file (including columns and extension)
    # Remove any columns not in test_df_orig except for the prediction column
    # If test_df_orig has 'label', we overwrite it; otherwise, we add it

    # 6. Save results
    results_ext = get_results_extension(TEST_CSV)
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME + results_ext)
    # Save with same columns as test, but with 'label' column replaced by prediction
    # If 'label' not in test_df_orig, add it as the last column
    if pred_col not in test_df_orig.columns:
        results_df = pd.concat([test_df_orig, malignancy_proba.rename(pred_col)], axis=1)
    else:
        results_df[pred_col] = malignancy_proba.values

    # Save in the same format as test file
    if results_ext == '.csv':
        results_df.to_csv(results_path, index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {results_ext}")

    # 7. Validation checks
    # a. Check number of rows
    assert len(results_df) == len(test_df_orig), "Number of predictions does not match number of test samples."
    # b. Check indices preserved
    assert all(results_df.index == test_df_orig.index), "Test data indices are not preserved in the results."
    # c. Check column names
    assert pred_col in results_df.columns, f"Prediction column '{pred_col}' missing from results."
    # d. Check output format
    assert os.path.exists(results_path), f"Results file was not saved at {results_path}."
    # e. Sanity check: predictions are floats between 0 and 1
    assert np.all((results_df[pred_col] >= 0) & (results_df[pred_col] <= 1)), "Predictions are not in [0, 1] range."

    print(f"Results saved to: {results_path}")

    # 8. Validation metric on held-out validation set
    try:
        val_pred_proba = predictor.predict_proba(val_data, as_pandas=True)
        if 1 in val_pred_proba.columns:
            val_malignancy_proba = val_pred_proba[1]
        elif '1' in val_pred_proba.columns:
            val_malignancy_proba = val_pred_proba['1']
        else:
            val_malignancy_proba = val_pred_proba.iloc[:, -1]
        val_true = val_data['label'].values
        val_auc = roc_auc_score(val_true, val_malignancy_proba)
        print(f"Validation AUROC: {val_auc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()