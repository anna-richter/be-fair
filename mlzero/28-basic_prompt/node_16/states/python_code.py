"""
Skin Lesion Malignancy Probability Prediction using AutoGluon Tabular

This script trains a tabular machine learning model using AutoGluon Tabular to predict the probability that a skin lesion is malignant, based on metadata and image references.
- Loads and preprocesses the data (removes NA labels, drops index columns).
- Trains a binary classifier to predict malignancy probability.
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions in the same format as the test file, with correct column names.
- Performs validation checks to ensure output correctness.

# Installation requirements (run before executing this script):
# pip install autogluon.tabular pandas scikit-learn

"""

import os
import random
import time
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/28-basic_prompt/node_16/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
# Find test CSV (must not be the training file)
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.endswith('.csv') and fname != "mydataset.csv":
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

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

    # 3. Model training
    # Prepare output model directory with random timestamp
    timestamp = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_tabular_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Select features: drop columns not useful for tabular prediction (e.g., image_name)
    # We'll use only metadata columns for tabular prediction
    feature_cols = [col for col in train_df.columns if col not in ['label', 'image_name']]
    # If all features are dropped, fallback to using image_name as a categorical feature
    if not feature_cols:
        feature_cols = ['image_name']

    # Train predictor
    predictor = TabularPredictor(
        label='label',
        problem_type='binary',
        eval_metric='roc_auc',
        path=model_dir
    )
    predictor.fit(
        train_df[feature_cols + ['label']],
        presets="extreme"
    )

    # 4. Prediction on test set
    # Ensure all feature columns exist in test set
    for col in feature_cols:
        if col not in test_df.columns:
            # If missing, fill with default value (e.g., NaN)
            test_df[col] = np.nan

    # Predict_proba returns probability for class 1 (malignant)
    test_pred_proba = predictor.predict_proba(test_df[feature_cols])
    # If output is DataFrame, get probability for class 1
    if isinstance(test_pred_proba, pd.DataFrame):
        if 1 in test_pred_proba.columns:
            malignancy_prob = test_pred_proba[1].values
        elif '1' in test_pred_proba.columns:
            malignancy_prob = test_pred_proba['1'].values
        else:
            malignancy_prob = test_pred_proba.iloc[:, -1].values
    else:
        malignancy_prob = np.array(test_pred_proba)

    # 5. Save predictions
    # Output format: same as test file, but with a column for malignancy probability
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

    # 6. Validation checks
    # a. Check output file has same number of rows as test set
    pred_df = pd.read_csv(results_path) if test_ext == ".csv" else (
        pd.read_parquet(results_path) if test_ext in [".parquet", ".pq"] else pd.read_csv(results_path, sep='\t')
    )
    assert len(pred_df) == len(test_df), "Prediction file row count does not match test set."
    # b. Check column names
    assert 'label' in pred_df.columns, "Output file missing 'label' column."
    # c. Check output format
    assert results_path.startswith(OUTPUT_DIR), "Results file not saved in output directory."
    # d. Check probabilities are in [0, 1]
    assert np.all((pred_df['label'] >= 0) & (pred_df['label'] <= 1)), "Predicted probabilities not in [0, 1]."

    print(f"Predictions saved to: {results_path}")
    print(f"Model saved to: {model_dir}")