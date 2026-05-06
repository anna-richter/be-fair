"""
Skin Lesion Malignancy Prediction Script

This script trains an image classification model using AutoGluon MultiModal on a curated skin lesion dataset.
It performs the following:
- Loads and preprocesses the data (removes NA labels, drops index column).
- Trains an image classification model to predict malignancy probability (0–1) for each lesion.
- Saves the trained model to a timestamped folder in the specified output directory.
- Makes predictions on the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation using a 10% holdout from the training data and prints AUROC.
- Includes validation checks to ensure output correctness.

# Installation (uncomment if needed):
# pip install autogluon.multimodal
# pip install pandas scikit-learn

Usage:
- Place this script in your working environment.
- Ensure the data folder and output directory exist and are accessible.
- Run the script as a standalone program.

Author: AutoML Agent
"""

import os
import random
import time
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Set paths
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/30-basic_prompt/node_9/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

# Find test CSV file (must exist)
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

IMAGE_ROOT = DATA_DIR  # Images are referenced by image_name column, and are in the same folder

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time())
    rand = random.randint(1000, 9999)
    folder = os.path.join(base_dir, f"model_{ts}_{rand}")
    return folder

def main():
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

    # 3. Prepare data for AutoGluon
    # AutoGluon expects an image path column; we'll create a new column with absolute image paths
    def image_path_fn(row):
        # Assume images are in DATA_DIR and have extensions (try .jpg, .png, .jpeg)
        base = os.path.join(IMAGE_ROOT, row['image_name'])
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        # If not found, try without extension (maybe image_name already has extension)
        if os.path.exists(base):
            return base
        raise FileNotFoundError(f"Image file not found for {row['image_name']}")
    train_df['image_path'] = train_df.apply(image_path_fn, axis=1)
    test_df['image_path'] = test_df.apply(image_path_fn, axis=1)

    # 4. Hold out 10% validation set from training data
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    # 5. Train model
    model_dir = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)

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
        presets='best_quality',
    )

    # 6. Save model (already saved by predictor.fit)

    # 7. Prediction on test set
    # Predict_proba returns probability for class 1 (malignant)
    test_pred_proba = predictor.predict_proba(test_df[['image_path']])
    # test_pred_proba is a DataFrame with columns [0, 1] for class probabilities
    # We want the probability for class 1 (malignant)
    if 1 in test_pred_proba.columns:
        malignancy_prob = test_pred_proba[1]
    else:
        # Sometimes columns are ['non-malignant', 'malignant']
        # Try to find the right column
        colnames = [str(c).lower() for c in test_pred_proba.columns]
        if 'malignant' in colnames:
            malignancy_prob = test_pred_proba[test_pred_proba.columns[colnames.index('malignant')]]
        else:
            raise ValueError("Could not find 'malignant' class in prediction columns.")

    # 8. Prepare output DataFrame
    # Output format: same as test file, but with the required prediction column(s)
    output_df = test_df.copy()
    output_df['label'] = malignancy_prob.values
    output_df = output_df[['label']]
    # Save with the same format and extension as test file
    test_ext = os.path.splitext(TEST_CSV)[1]
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    output_df.to_csv(result_path, index=True if test_df.index.name is not None else False)

    # 9. Validation step: compute AUROC on held-out validation set
    try:
        val_pred_proba = predictor.predict_proba(val_data[['image_path']])
        if 1 in val_pred_proba.columns:
            val_prob = val_pred_proba[1]
        else:
            colnames = [str(c).lower() for c in val_pred_proba.columns]
            if 'malignant' in colnames:
                val_prob = val_pred_proba[val_pred_proba.columns[colnames.index('malignant')]]
            else:
                raise ValueError("Could not find 'malignant' class in validation prediction columns.")
        val_score = roc_auc_score(val_data['label'], val_prob)
        print(f"Validation AUROC: {val_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 10. Validation checks
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

if __name__ == "__main__":
    main()