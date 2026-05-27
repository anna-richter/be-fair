"""
Skin Lesion Image Binary Classification with AutoGluon MultiModal

This script trains an AutoGluon MultiModal image classifier to predict malignancy probability (0-1) for skin lesion images.
It:
- Loads and preprocesses training data (removes NA labels, drops index column)
- Trains a binary classifier using image and tabular features
- Saves the trained model to a timestamped folder in the output directory
- Predicts malignancy probability for each test image, preserving original indices and output format
- Saves predictions in the same format as test.csv to the output directory
- Performs validation (10% holdout) and prints AUROC if possible
- Performs strict validation checks on the output file

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Place this script in any location and run it. It expects the following data structure:
      - /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/train.csv
      - /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv
      - /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/MyImages/*.jpg

Output:
    - Model folder: /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/output/<timestamp>_automm_model/
    - Prediction file: /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/output/results.csv

Author: AutoML Agent
"""

# Installation steps (uncomment if running in a fresh environment)
# !pip install --upgrade pip
# !pip install autogluon.multimodal

import os
import sys
import time
import uuid
import shutil
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Paths
    DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"
    IMG_DIR = os.path.join(DATA_DIR, "MyImages")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data Loading and Preprocessing
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples without valid labels (drop NA in 'label')
    train = train.dropna(subset=['label'])

    # Map label to binary: 'malignant' = 1, others (e.g., 'non-neoplastic') = 0
    # If there are only two classes, this is safe.
    label_col = 'label'
    positive_label = 'malignant'
    train[label_col] = (train[label_col].str.lower() == positive_label).astype(int)

    # Add absolute image path columns for AutoGluon
    def image_name_to_path(image_name):
        # Some image_name may have extension, some may not
        if not image_name.endswith('.jpg'):
            image_name = image_name + '.jpg'
        return os.path.join(IMG_DIR, image_name)

    train['image'] = train['image_name'].apply(image_name_to_path)
    test['image'] = test['image_name'].apply(image_name_to_path)

    # 2. Validation Split (10% holdout if no validation set is given)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train,
        test_size=0.1,
        random_state=42,
        stratify=train[label_col]
    )

    # 3. Model Training
    # Save model to a timestamped folder in output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    model_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_automm_model")
    os.makedirs(model_dir, exist_ok=True)

    # Use all features except label and image_name as input
    # (image, skin_tone, alternative_skin_tone, expert_opinion)
    # expert_opinion is mostly NaN, but AutoGluon can handle missing values

    # Prepare training data
    feature_cols = ['image', 'skin_tone', 'alternative_skin_tone', 'expert_opinion']
    # Remove columns not needed for training
    train_data_for_fit = train_data[feature_cols + [label_col]].copy()

    # Model
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_dir
    )

    # Fit model (presets are already configured, no need for HPO)
    predictor.fit(
        train_data=train_data_for_fit,
        time_limit=1800,  # 30 min max for training
        # hyperparameters can be left as default for binary image classification
    )

    # Save model (already saved in model_dir by AutoGluon)

    # 4. Prediction on Test Set
    # Prepare test data for prediction (must include all features used in training except label)
    test_for_pred = test.copy()
    test_for_pred['image'] = test_for_pred['image_name'].apply(image_name_to_path)
    # For missing tabular columns, fill with NaN (AutoGluon handles missing)
    for col in ['skin_tone', 'alternative_skin_tone', 'expert_opinion']:
        if col not in test_for_pred.columns:
            test_for_pred[col] = np.nan

    # Ensure columns order matches training
    test_for_pred = test_for_pred[['image', 'skin_tone', 'alternative_skin_tone', 'expert_opinion', 'image_name']]

    # Predict probabilities (malignancy probability)
    proba = predictor.predict_proba(test_for_pred[feature_cols])
    # proba is a DataFrame with columns [0, 1] (class 0, class 1)
    # We want the probability of class 1 (malignant)
    if 1 in proba.columns:
        malignancy_prob = proba[1].values
    elif '1' in proba.columns:
        malignancy_prob = proba['1'].values
    else:
        # fallback: take the last column
        malignancy_prob = proba.iloc[:, -1].values

    # Prepare output DataFrame
    # Output format and extension must match test.csv (CSV)
    # Output columns: all columns from test.csv + one column for malignancy probability
    # Since test.csv has columns ['image_name'], we will output ['image_name', 'malignancy_probability']
    output_df = test[['image_name']].copy()
    output_df['malignancy_probability'] = malignancy_prob

    # Save to output directory as "results.csv"
    output_path = os.path.join(OUTPUT_DIR, "results.csv")
    output_df.to_csv(output_path, index=False)

    # 5. Validation (AUROC on holdout set)
    try:
        val_data_for_pred = val_data[feature_cols].copy()
        val_true = val_data[label_col].values
        val_proba = predictor.predict_proba(val_data_for_pred)
        if 1 in val_proba.columns:
            val_pred = val_proba[1].values
        elif '1' in val_proba.columns:
            val_pred = val_proba['1'].values
        else:
            val_pred = val_proba.iloc[:, -1].values
        val_auroc = roc_auc_score(val_true, val_pred)
        print(f"Validation AUROC: {val_auroc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 6. Output Validation Checks
    # a) Check output file exists
    assert os.path.isfile(output_path), f"Prediction file not found at {output_path}"

    # b) Check number of rows matches test set
    pred_df = pd.read_csv(output_path)
    assert len(pred_df) == len(test), f"Prediction file has {len(pred_df)} rows, expected {len(test)}"

    # c) Check indices correspond to test set (image_name order)
    assert all(pred_df['image_name'].values == test['image_name'].values), \
        "Prediction file image_name order does not match test set"

    # d) Check output columns
    expected_cols = ['image_name', 'malignancy_probability']
    assert list(pred_df.columns) == expected_cols, \
        f"Prediction file columns {list(pred_df.columns)} do not match expected {expected_cols}"

    # e) Check output format (CSV)
    assert output_path.endswith('.csv'), "Output file must be CSV"

    # f) Check malignancy_probability is float in [0,1]
    assert np.all((pred_df['malignancy_probability'] >= 0) & (pred_df['malignancy_probability'] <= 1)), \
        "Malignancy probabilities are not all in [0,1]"

    print("All validation checks passed. Results saved to:", output_path)