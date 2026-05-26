"""
Skin Lesion Malignancy Probability Prediction using AutoGluon MultiModal

This script:
- Loads and preprocesses image classification data for skin lesion malignancy prediction.
- Trains an AutoGluon MultiModalPredictor on RGB JPEG images and available tabular metadata.
- Uses a strong image backbone and optimized training schedule for improved performance.
- Predicts malignancy probability (float in [0,1]) for each test image, preserving original indices.
- Saves the trained model and prediction results to the specified output directory.
- Performs validation (AUROC) on a held-out validation set if training labels are available.
- Includes robust validation checks for output format, indices, and prediction sanity.

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Ensure the data folder and image files are present as described.
    Run this script as a standalone Python file.

Author: AutoML Agent
"""

# Installation steps (uncomment if running in a fresh environment)
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal

import os
import random
import time
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Constants
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/32-basic_prompt/node_18/output"

# Output file name will match test file extension
RESULTS_BASENAME = "results"

def get_random_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time()) + random.randint(0, 100000)
    folder = os.path.join(base_dir, f"autogluon_model_{ts}")
    return folder

def map_label_to_binary(label):
    """Map string label to binary: 'malignant'->1, others->0."""
    return 1 if str(label).strip().lower() == "malignant" else 0

def get_image_path(image_name):
    """Get absolute path to image file given image_name (without extension)."""
    return os.path.join(IMG_DIR, f"{image_name}.jpg")

def main():
    # 1. Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # 2. Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # 3. Remove training samples without valid labels (drop NA in 'label')
    train = train.dropna(subset=['label']).reset_index(drop=True)

    # 4. Map label to binary (malignant=1, else=0)
    train['label'] = train['label'].map(map_label_to_binary)

    # 5. Add absolute image path column for train/test
    train['image'] = train['image_name'].apply(get_image_path)
    test['image'] = test['image_name'].apply(get_image_path)

    # 6. Only use tabular features present in BOTH train and test
    # This avoids KeyError if test set lacks tabular columns
    possible_tabular_cols = ['skin_tone', 'alternative_skin_tone']
    tabular_cols = [col for col in possible_tabular_cols if col in train.columns and col in test.columns]
    # expert_opinion is always NaN, so we drop it
    if 'expert_opinion' in train.columns:
        train = train.drop(columns=['expert_opinion'])
    if 'expert_opinion' in test.columns:
        test = test.drop(columns=['expert_opinion'])

    # 7. Validation split (10% holdout if no validation set is provided)
    VALIDATION_RATIO = 0.1
    np.random.seed(42)
    val_idx = np.random.choice(train.index, size=int(len(train) * VALIDATION_RATIO), replace=False)
    val_data = train.loc[val_idx].reset_index(drop=True)
    train_data = train.drop(index=val_idx).reset_index(drop=True)

    # 8. Model training
    model_dir = get_random_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)

    # Prepare train/val/test for AutoGluon: only use columns that exist in each
    # Always include 'image' and 'label' for train/val, 'image' for test
    train_cols = ['image', 'label'] + tabular_cols
    val_cols = ['image', 'label'] + tabular_cols
    test_cols = ['image'] + tabular_cols

    train_data_ag = train_data[train_cols].copy()
    val_data_ag = val_data[val_cols].copy()
    test_data_ag = test[test_cols].copy()

    # Model architecture improvements: use a strong image backbone and optimized training schedule
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )

    # Use a strong image backbone and correct hyperparameter keys (optim.*)
    predictor.fit(
        train_data=train_data_ag,
        tuning_data=val_data_ag,
        presets="best_quality",
        hyperparameters={
            "model.timm_image.checkpoint_name": "convnext_base_in22ft1k",  # Stronger backbone
            "env.num_gpus": 1,
            "env.per_gpu_batch_size": 16,  # Reasonable batch size for 1 GPU
            "env.num_workers": 8,
            "optim.max_epochs": 22,  # More epochs for better performance
            "optim.lr": 2e-4,
            "optim.lr_schedule": "cosine_decay",
        },
        time_limit=2800  # Leave time for inference and saving
    )

    # 9. Save model (already saved in model_dir by AutoGluon)

    # 10. Prediction on test set
    # Prepare test data for prediction (must match training features)
    test_pred_df = test_data_ag.copy()

    # Predict probabilities for class 1 (malignant)
    proba = predictor.predict_proba(test_pred_df)
    # For binary, predict_proba returns a DataFrame with columns [0,1]
    # We want the probability for class 1 (malignant)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            malignancy_prob = proba[1].values
        elif "1" in proba.columns:
            malignancy_prob = proba["1"].values
        else:
            # Fallback: take the last column (should be class 1)
            malignancy_prob = proba.iloc[:, -1].values
    else:
        # Should not happen, but fallback
        malignancy_prob = np.array(proba)

    # 11. Prepare output DataFrame
    # Output must have same indices and row order as test.csv
    # Output column name: 'label' (as in train.csv)
    results_df = test[['image_name']].copy()
    results_df['label'] = malignancy_prob

    # 12. Save results in same format/extension as test.csv
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + test_ext)
    if test_ext == ".csv":
        results_df.to_csv(results_path, index=False)
    elif test_ext in [".tsv", ".txt"]:
        results_df.to_csv(results_path, sep='\t', index=False)
    else:
        raise ValueError(f"Unknown test file extension: {test_ext}")

    # 13. Validation checks
    # a) Indices: output must have same number of rows and order as test
    assert len(results_df) == len(test), "Prediction file row count does not match test set."
    assert all(results_df['image_name'].values == test['image_name'].values), "Prediction file row order does not match test set."
    # b) Output columns: must be ['image_name', 'label']
    assert list(results_df.columns) == ['image_name', 'label'], f"Output columns {list(results_df.columns)} do not match required ['image_name', 'label']"
    # c) Output format: extension matches test file
    assert os.path.splitext(results_path)[1].lower() == test_ext, "Output file extension does not match test file."
    # d) Prediction values: floats in [0,1]
    assert np.all((results_df['label'] >= 0) & (results_df['label'] <= 1)), "Predicted probabilities are not in [0,1]."

    # 14. Validation metric on held-out validation set
    try:
        val_pred_proba = predictor.predict_proba(val_data_ag)
        if isinstance(val_pred_proba, pd.DataFrame):
            if 1 in val_pred_proba.columns:
                val_probs = val_pred_proba[1].values
            elif "1" in val_pred_proba.columns:
                val_probs = val_pred_proba["1"].values
            else:
                val_probs = val_pred_proba.iloc[:, -1].values
        else:
            val_probs = np.array(val_pred_proba)
        val_auc = roc_auc_score(val_data_ag['label'], val_probs)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")

if __name__ == "__main__":
    main()