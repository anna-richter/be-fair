"""
Skin Lesion Malignancy Prediction Script

This script trains an image classification model using AutoGluon MultiModal to predict the probability
that a skin lesion is malignant, based on images referenced in a CSV file. It performs the following:

- Loads and preprocesses the data (removes NA labels, drops index column).
- Trains a binary image classification model (malignant vs. benign).
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Provides a function to predict malignancy probabilities for a folder of new images.
- Performs validation checks to ensure output correctness.

Installation requirements (run before executing this script):
# pip install autogluon.multimodal
# pip install pandas numpy

Author: AutoML Agent
"""

import os
import random
import time
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor

# Set random seed for reproducibility
RANDOM_SEED = 42

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/28-basic_prompt/node_3/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
# Assume test file is named 'test.csv' in the same folder (adjust if needed)
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMAGE_FOLDER = DATA_DIR  # Images are referenced by image_name in the CSV

# Output file name (format/extension must match test data file)
RESULTS_BASENAME = "results"

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time() * 1000) + random.randint(0, 9999)
    folder = os.path.join(base_dir, f"model_{ts}")
    return folder

def get_image_path(row):
    """Helper to get absolute image path from image_name."""
    return os.path.join(IMAGE_FOLDER, row['image_name'])

def prepare_data(df):
    """Drop NA labels and unnecessary index column from training data."""
    # Drop rows where label is NA
    df = df.dropna(subset=['label'])
    # Drop index column if present
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
    return df

def encode_label(df):
    """Convert 'label' column to binary: 1 for malignant, 0 for non-malignant."""
    # Lowercase for robustness
    df['label'] = df['label'].str.lower()
    df['label'] = (df['label'] == 'malignant').astype(int)
    return df

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    # Remove NA labels and index column from training data only
    train_df = prepare_data(train_df)
    train_df = encode_label(train_df)
    # Add absolute image path column for AutoGluon
    train_df['image_path'] = train_df.apply(get_image_path, axis=1)

    # 2. Prepare test data
    # Find test file (must match extension/format of train)
    # Try common test file names if not found
    if not os.path.exists(TEST_CSV):
        # Try to find a test file in the data directory
        found = False
        for fname in os.listdir(DATA_DIR):
            if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
                TEST_CSV = os.path.join(DATA_DIR, fname)
                found = True
                break
        if not found:
            raise FileNotFoundError("Test CSV file not found in data directory.")
    test_df = pd.read_csv(TEST_CSV)
    test_index = test_df.index.copy()
    # Drop index column if present (but DO NOT drop any rows)
    for idx_col in ['Unnamed: 0', 'index']:
        if idx_col in test_df.columns:
            test_df = test_df.drop(columns=[idx_col])
    # Add absolute image path column for AutoGluon
    test_df['image_path'] = test_df.apply(get_image_path, axis=1)

    # 3. Train model
    # Use AUROC as eval metric for binary classification
    model_save_path = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_save_path, exist_ok=True)
    predictor = MultiModalPredictor(
        label='label',
        problem_type='binary',
        eval_metric='roc_auc',
        path=model_save_path,
        seed=RANDOM_SEED,
    )
    # Only use image_path and label for training
    train_data = train_df[['image_path', 'label']]
    predictor.fit(
        train_data,
        presets='best_quality',
        time_limit=None,
    )

    # 4. Predict on test set
    # Only use image_path for prediction
    test_data = test_df[['image_path']].copy()
    # Get probability for class '1' (malignant)
    y_pred_proba = predictor.predict_proba(test_data)
    # y_pred_proba is a DataFrame with columns [0, 1] (for each class)
    # We want the probability for class 1 (malignant)
    if 1 in y_pred_proba.columns:
        malignancy_prob = y_pred_proba[1]
    elif '1' in y_pred_proba.columns:
        malignancy_prob = y_pred_proba['1']
    else:
        raise RuntimeError("Could not find class 1 probability in prediction output.")

    # Prepare output DataFrame
    # Output column name must match training or sample submission
    # We'll use 'label' as the output column (since that's the target in train)
    results_df = pd.DataFrame({'label': malignancy_prob})
    # Restore original test indices
    results_df.index = test_index

    # 5. Save results in the same format/extension as test file
    # Determine extension
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + test_ext)
    if test_ext == '.csv':
        results_df.to_csv(results_path, index=True)  # preserve index
    elif test_ext in ['.parquet', '.pq']:
        results_df.to_parquet(results_path, index=True)
    elif test_ext in ['.xlsx', '.xls']:
        results_df.to_excel(results_path, index=True)
    else:
        raise ValueError(f"Unsupported test file extension: {test_ext}")

    # 6. Validation checks
    # a) Check number of rows matches test set
    assert len(results_df) == len(test_df), "Number of predictions does not match number of test samples."
    # b) Check indices match
    assert all(results_df.index == test_index), "Prediction indices do not match test data indices."
    # c) Check output column name
    assert list(results_df.columns) == ['label'], "Output column name does not match requirements."
    # d) Check output format
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
    # e) Check probabilities are in [0, 1]
    assert np.all((results_df['label'] >= 0) & (results_df['label'] <= 1)), "Predicted probabilities are not in [0, 1]."

    print(f"Model trained and saved to: {model_save_path}")
    print(f"Predictions saved to: {results_path}")

    # 7. Provide function for new image folder prediction
    def predict_folder(image_folder_path):
        """
        Given a folder path containing images, returns a DataFrame with image file names and
        malignancy probabilities (float in [0, 1]) for each image.
        """
        # List all image files in the folder (common image extensions)
        image_files = [f for f in os.listdir(image_folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        if not image_files:
            raise ValueError("No image files found in the provided folder.")
        # Prepare DataFrame for prediction
        df = pd.DataFrame({'image_name': image_files})
        df['image_path'] = df['image_name'].apply(lambda x: os.path.join(image_folder_path, x))
        # Predict probabilities
        y_proba = predictor.predict_proba(df[['image_path']])
        if 1 in y_proba.columns:
            malignancy_prob = y_proba[1]
        elif '1' in y_proba.columns:
            malignancy_prob = y_proba['1']
        else:
            raise RuntimeError("Could not find class 1 probability in prediction output.")
        result = pd.DataFrame({
            'image_name': image_files,
            'malignancy_probability': malignancy_prob.values
        })
        return result

    # Save the function for user access (optional: can be imported if this script is used as a module)
    globals()['predict_folder'] = predict_folder