"""
Skin Lesion Malignancy Probability Prediction using AutoGluon Tabular

This script trains a tabular machine learning model using AutoGluon Tabular to predict the probability
that a skin lesion is malignant, based on structured metadata (not image pixels).
It:
- Loads and preprocesses the data (removes NA labels, drops index columns).
- Trains a classifier using AutoGluon Tabular with presets="extreme".
- Saves the trained model to a timestamped folder in the output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation using AutoGluon's built-in validation and prints the best validation score.
- Includes validation checks to ensure output correctness.

# Installation (uncomment and run in bash if needed):
# pip install autogluon.tabular pandas

Author: AutoML Agent
"""

import os
import random
import time
import pandas as pd
from autogluon.tabular import TabularPredictor

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/30-basic_prompt/node_17/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

# Find test CSV file (must exist)
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time())
    rand = random.randint(1000, 9999)
    folder = os.path.join(base_dir, f"model_{ts}_{rand}")
    return folder

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Data preprocessing
    # Remove index column if present
    index_cols = [col for col in train_df.columns if col.lower() in ['unnamed: 0', 'index']]
    if index_cols:
        train_df = train_df.drop(columns=index_cols)
    index_cols_test = [col for col in test_df.columns if col.lower() in ['unnamed: 0', 'index']]
    if index_cols_test:
        test_df = test_df.drop(columns=index_cols_test)

    # Remove training samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])

    # Map label to binary: malignant=1, else 0
    train_df['label'] = (train_df['label'].str.lower() == 'malignant').astype(int)

    # For test set, keep all rows, do not drop any rows

    # 3. Prepare features for tabular model
    # Drop image_name (since we can't use image pixels in TabularPredictor)
    drop_cols = ['image_name']
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols, errors='ignore')  # errors='ignore' in case test set lacks this column

    # 4. Model training with AutoGluon Tabular
    model_dir = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)
    predictor = TabularPredictor(
        label='label',
        path=model_dir,
        problem_type='binary',
        eval_metric='roc_auc'
    )
    predictor.fit(
        train_data=X_train,
        presets="extreme"
    )

    # 5. Prediction on test set
    # Predict_proba returns probability for class 1 (malignant)
    test_pred_proba = predictor.predict_proba(X_test)
    # test_pred_proba is a DataFrame with columns [0, 1] for class probabilities
    # We want the probability for class 1 (malignant)
    if 1 in test_pred_proba.columns:
        malignancy_prob = test_pred_proba[1]
    else:
        # Sometimes columns are ['non-malignant', 'malignant']
        colnames = [str(c).lower() for c in test_pred_proba.columns]
        if 'malignant' in colnames:
            malignancy_prob = test_pred_proba[test_pred_proba.columns[colnames.index('malignant')]]
        else:
            raise ValueError("Could not find 'malignant' class in prediction columns.")

    # 6. Prepare output DataFrame
    output_df = test_df.copy()
    output_df['label'] = malignancy_prob.values
    output_df = output_df[['label']]
    # Save with the same format and extension as test file
    test_ext = os.path.splitext(TEST_CSV)[1]
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    output_df.to_csv(result_path, index=True if test_df.index.name is not None else False)

    # 7. Validation step: print best validation score from AutoGluon
    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_score = leaderboard['score_val'].iloc[0]
        print(f"Best AutoGluon validation AUROC: {best_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 8. Validation checks
    # a) Prediction file maintains exact test data indices
    pred_df = pd.read_csv(result_path, index_col=0 if test_df.index.name is not None else None)
    assert len(pred_df) == len(test_df), "Number of predictions does not match number of test samples."
    # b) Output column names match requirements
    assert list(pred_df.columns) == ['label'], f"Output columns {list(pred_df.columns)} do not match required ['label']."
    # c) Output format matches test file extension
    assert os.path.splitext(result_path)[1] == test_ext, "Output file extension does not match test file."
    # d) Sanity check: predictions are between 0 and 1
    assert ((pred_df['label'] >= 0) & (pred_df['label'] <= 1)).all(), "Predicted probabilities are not in [0, 1]."
    print("All validation checks passed. Results saved to:", result_path)