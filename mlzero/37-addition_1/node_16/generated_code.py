"""
Skin Lesion Malignancy Probability Prediction using AutoGluon Tabular

This script:
- Loads and preprocesses tabular metadata for skin lesion images.
- Trains an AutoGluon Tabular predictor to classify lesions as malignant or benign using metadata (not image pixels).
- Saves the trained model to a timestamped folder in the specified output directory.
- Makes predictions (malignancy probability) on the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation using AutoGluon's internal validation and prints the best validation score.
- Includes validation checks to ensure output correctness.

# Installation (if needed):
# !pip install autogluon.tabular pandas scikit-learn

Data paths and output directory are hardcoded as per instructions.
"""

import os
import random
import time
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

# ==== CONFIGURATION ====
DATA_DIR = "/home/anri21/be-fair/mlzero/addition_1_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/37-addition_1/node_16/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

def get_test_csv_path():
    # Try to find the test file in the data directory
    for fname in os.listdir(DATA_DIR):
        if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
            return os.path.join(DATA_DIR, fname)
    raise FileNotFoundError("Test CSV file not found in data directory.")

def make_output_model_dir():
    # Create a random timestamped model directory in OUTPUT_DIR
    ts = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_model_{ts}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

if __name__ == "__main__":
    # ==== 1. Load and preprocess data ====
    train_df = pd.read_csv(TRAIN_CSV)
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    # Remove training samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])
    train_df = train_df.reset_index(drop=True)

    # Map 'label' to binary: malignant=1, benign=0 (assuming 'malignant' and 'benign' are possible values)
    # If 'non-neoplastic' is present, treat as benign (0)
    label_map = {'malignant': 1, 'benign': 0, 'non-neoplastic': 0}
    train_df['label'] = train_df['label'].map(label_map)
    if train_df['label'].isnull().any():
        raise ValueError("Unknown label values found in training data.")

    # ==== 2. Prepare test data ====
    test_csv_path = get_test_csv_path()
    test_df = pd.read_csv(test_csv_path)
    test_index = test_df.index.copy()  # Save original indices for validation

    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])

    # Find common columns (excluding label)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = list((train_cols & test_cols) - {'label'})

    # ==== 3. Train model ====
    model_dir = make_output_model_dir()
    predictor = TabularPredictor(
        label='label',
        path=model_dir,
        problem_type='binary',
        eval_metric='roc_auc'
    )
    # Use all available metadata columns for training
    predictor.fit(
        train_df[common_cols + ['label']],
        presets="extreme"
    )

    # ==== 4. Predict on test set ====
    # Predict_proba returns probability for each class; for binary, column 1 is '1' (malignant)
    y_pred_proba = predictor.predict_proba(test_df[common_cols])
    # If output is DataFrame, get probability for class '1'
    if isinstance(y_pred_proba, pd.DataFrame):
        if 1 in y_pred_proba.columns:
            malignancy_prob = y_pred_proba[1].values
        elif '1' in y_pred_proba.columns:
            malignancy_prob = y_pred_proba['1'].values
        else:
            # fallback: take the second column
            malignancy_prob = y_pred_proba.iloc[:, 1].values
    else:
        # If output is ndarray
        malignancy_prob = y_pred_proba[:, 1]

    # ==== 5. Prepare output ====
    output_df = test_df.copy()
    output_df['label'] = malignancy_prob

    # Save output in same format as test file (CSV)
    test_ext = os.path.splitext(test_csv_path)[1]
    results_path = os.path.join(OUTPUT_DIR, "results" + test_ext)
    output_df.to_csv(results_path, index=False)

    # ==== 6. Validation checks ====
    # 1. Check number of rows matches test set
    assert len(output_df) == len(test_df), "Number of predictions does not match number of test samples."
    # 2. Check indices preserved
    assert all(output_df.index == test_index), "Test indices not preserved in output."
    # 3. Check output columns: must include 'label'
    assert 'label' in output_df.columns, "Output file missing 'label' column."
    # 4. Check output format: extension matches test file
    assert os.path.splitext(results_path)[1] == test_ext, "Output file extension does not match test file."
    # 5. Check predictions are floats in [0,1]
    assert np.all((output_df['label'] >= 0) & (output_df['label'] <= 1)), "Predicted probabilities not in [0,1]."

    # ==== 7. Validation on holdout set ====
    # AutoGluon Tabular automatically reports validation scores; print best validation score
    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_score = leaderboard['score_val'].iloc[0]
        print(f"Best AutoGluon internal validation AUROC: {best_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")