"""
Skin Lesion Malignancy Probability Prediction using AutoGluon Tabular

This script trains a tabular machine learning model using AutoGluon Tabular to predict the probability
that a skin lesion is malignant, based on tabular metadata and image references. It performs the following:

- Loads and preprocesses the data (removes NA labels, drops index column).
- Trains a binary classifier (malignant vs. benign) using tabular features.
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Provides a function to predict malignancy probabilities for a folder of new images (using metadata).
- Performs validation checks to ensure output correctness.

Installation requirements (run before executing this script):
# pip install autogluon.tabular pandas numpy

Author: AutoML Agent
"""

# =======================
# Installation (run in bash before running this script):
# pip install autogluon.tabular pandas numpy
# =======================

import os
import time
import random
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

# Set random seed for reproducibility
RANDOM_SEED = 42

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/28-basic_prompt/node_2/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
# Assume test file is named 'test.csv' in the same folder (adjust if needed)
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

RESULTS_BASENAME = "results"

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time() * 1000) + random.randint(0, 9999)
    folder = os.path.join(base_dir, f"model_{ts}")
    return folder

def prepare_data(df):
    """Drop NA labels and unnecessary index column from training data."""
    df = df.dropna(subset=['label'])
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    return df

def encode_label(df):
    """Convert 'label' column to binary: 1 for malignant, 0 for non-malignant."""
    df['label'] = df['label'].str.lower()
    df['label'] = (df['label'] == 'malignant').astype(int)
    return df

def main():
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = prepare_data(train_df)
    train_df = encode_label(train_df)

    # 2. Prepare test data
    global TEST_CSV
    if not os.path.exists(TEST_CSV):
        # Try to find a test file in the data directory
        for fname in os.listdir(DATA_DIR):
            if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
                TEST_CSV = os.path.join(DATA_DIR, fname)
                break
        else:
            raise FileNotFoundError("Test CSV file not found in data directory.")
    test_df = pd.read_csv(TEST_CSV)
    test_index = test_df.index.copy()
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in test_df.columns:
            test_df = test_df.drop(columns=[idx_col])

    # 3. Select features for training (exclude image_name, expert_opinion, and label)
    # Use only tabular features
    feature_cols = [col for col in train_df.columns if col not in ['label', 'image_name', 'expert_opinion']]
    # If expert_opinion is useful, you can include it, but it's NaN in sample data

    # 4. Train model
    model_save_path = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_save_path, exist_ok=True)
    predictor = TabularPredictor(
        label='label',
        path=model_save_path,
        problem_type='binary',
        eval_metric='roc_auc'
    )
    predictor.fit(
        train_df[feature_cols + ['label']],
        presets="extreme",
        time_limit=None,
        verbosity=2
    )

    # 5. Predict on test set (malignancy probability)
    # Ensure test_df has all required features
    test_features = test_df[feature_cols].copy()
    y_pred_proba = predictor.predict_proba(test_features)
    # y_pred_proba is a DataFrame with columns [0, 1] (for each class)
    # We want the probability for class 1 (malignant)
    if 1 in y_pred_proba.columns:
        malignancy_prob = y_pred_proba[1]
    else:
        # Sometimes columns are strings
        malignancy_prob = y_pred_proba['1']

    results_df = pd.DataFrame({'label': malignancy_prob})
    results_df.index = test_index

    # 6. Save results in the same format/extension as test file
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + test_ext)
    if test_ext == '.csv':
        results_df.to_csv(results_path, index=True)
    elif test_ext in ['.parquet', '.pq']:
        results_df.to_parquet(results_path, index=True)
    elif test_ext in ['.xlsx', '.xls']:
        results_df.to_excel(results_path, index=True)
    else:
        raise ValueError(f"Unsupported test file extension: {test_ext}")

    # 7. Validation checks
    assert len(results_df) == len(test_df), "Number of predictions does not match number of test samples."
    assert all(results_df.index == test_index), "Prediction indices do not match test data indices."
    assert list(results_df.columns) == ['label'], "Output column name does not match requirements."
    if test_ext == '.csv':
        check_df = pd.read_csv(results_path, index_col=0)
    elif test_ext in ['.parquet', '.pq']:
        check_df = pd.read_parquet(results_path)
    elif test_ext in ['.xlsx', '.xls']:
        check_df = pd.read_excel(results_path, index_col=0)
    else:
        check_df = None
    if check_df is not None:
        assert len(check_df) == len(test_df), "Saved prediction file row count mismatch."
        assert list(check_df.columns) == ['label'], "Saved prediction file column mismatch."
    assert np.all((results_df['label'] >= 0) & (results_df['label'] <= 1)), "Predicted probabilities are not in [0, 1]."

    print(f"Model trained and saved to: {model_save_path}")
    print(f"Predictions saved to: {results_path}")

    # 8. Provide function for new image folder prediction (using metadata)
    def predict_folder(metadata_csv_path):
        """
        Given a CSV file containing metadata for new images (with same columns as training features),
        returns a DataFrame with malignancy probabilities (float in [0, 1]) for each row.
        """
        meta_df = pd.read_csv(metadata_csv_path)
        for idx_col in ['Unnamed: 0', 'index']:
            if idx_col in meta_df.columns:
                meta_df = meta_df.drop(columns=[idx_col])
        meta_features = meta_df[feature_cols].copy()
        y_proba = predictor.predict_proba(meta_features)
        if 1 in y_proba.columns:
            malignancy_prob = y_proba[1]
        else:
            malignancy_prob = y_proba['1']
        result = pd.DataFrame({
            'malignancy_probability': malignancy_prob
        }, index=meta_df.index)
        return result

    # Save the function for user access (optional: can be imported if this script is used as a module)
    globals()['predict_folder'] = predict_folder

if __name__ == "__main__":
    main()