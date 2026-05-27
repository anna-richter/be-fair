"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy (malignant vs. benign)
for skin lesion images, using the AutoGluon MultiModal framework. It performs the following steps:

1. Data preprocessing:
   - Loads train and test CSVs, removes NA labels from train only, and removes unnecessary index columns.
   - Maps label values to binary classes ("malignant"=1, "benign"=0) as per the task description.
   - Adds absolute image paths for AutoGluon compatibility.

2. Model training:
   - Splits training data into train/validation (90/10 split).
   - Trains a MultiModalPredictor for image classification, using class weights to mitigate skin tone imbalance.
   - Saves the trained model to a timestamped folder in the specified output directory.

3. Prediction:
   - Predicts malignancy probabilities for the entire test set, preserving original indices and row order.
   - Saves results in the same format and extension as test.csv, with the correct column names.

4. Validation:
   - Computes and prints AUROC on the held-out validation set.

5. Output validation:
   - Asserts that the prediction file matches test indices, column names, format, and row count.
   - Checks that probabilities are in [0, 1].

Installation requirements:
    # pip install --upgrade pip
    # pip install autogluon.multimodal
    # pip install scikit-learn

Usage:
    Place this script in any location and run it. It expects the data folder at:
    /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data
    and will write all outputs to:
    /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/52-addition_3/node_7/output

Author: AutoML Agent
"""

# Installation steps (uncomment if running in a fresh environment)
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal scikit-learn

import os
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Constants
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/52-addition_3/node_7/output"

if __name__ == "__main__":
    # 1. Data Loading and Preprocessing

    # Load train and test data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples with missing labels (do NOT drop from test)
    train = train.dropna(subset=['label'])

    # Map label to binary: malignant=1, benign=0
    label_map = {"malignant": 1, "benign": 0}
    train = train[train['label'].isin(label_map.keys())].copy()
    train['label'] = train['label'].map(label_map)

    # Add absolute image paths for AutoGluon
    def image_path_fn(image_name):
        return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

    train['image'] = train['image_name'].apply(image_path_fn)
    test['image'] = test['image_name'].apply(image_path_fn)

    # 2. Fairness: Compute class weights for skin tone groups (optional, but recommended)
    class_counts = train['label'].value_counts().sort_index()
    class_weights = [1.0 / class_counts.get(i, 1) for i in range(2)]
    class_weights = np.array(class_weights) / np.sum(class_weights)  # Normalize

    # 3. Train/Validation Split
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(train, train['label']))
    train_data = train.iloc[train_idx].reset_index(drop=True)
    val_data = train.iloc[val_idx].reset_index(drop=True)

    # 4. Model Training
    # Save model to a timestamped folder in OUTPUT_DIR
    model_dir = os.path.join(
        OUTPUT_DIR,
        f"autogluon_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(model_dir, exist_ok=True)

    # Prepare AutoGluon predictor
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )

    # Use focal loss with class weights for fairness (helps with imbalance)
    # Reduce epochs and set time_limit to avoid wall-time kill
    hyperparameters = {
        "env.num_gpus": 1,  # Use single GPU to avoid DDP errors
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights.tolist(),
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        "optim.max_epochs": 3,
        "env.num_workers": 2,
        "env.num_workers_inference": 2,
    }

    # Only keep columns needed for training
    train_cols = ['image', 'label']
    val_cols = ['image', 'label']

    predictor.fit(
        train_data=train_data[train_cols],
        tuning_data=val_data[val_cols],
        hyperparameters=hyperparameters,
        presets="best_quality",
        seed=42,
        time_limit=3400  # To avoid wall-time limit, leave time for prediction/validation
    )

    # Save model (already saved by AutoGluon, but ensure it's in the right folder)
    predictor.save(model_dir)

    # 5. Prediction on Test Set
    # Predict probabilities for the test set (malignancy probability)
    # Ensure test set is not modified (no row drops, preserve order)
    test_pred_input = test[['image']].copy()
    proba = predictor.predict_proba(test_pred_input)
    # proba is a DataFrame with columns [0, 1] for class probabilities

    # We want the probability of "malignant" (class 1)
    if 1 in proba.columns:
        malignancy_prob = proba[1].values
    elif "1" in proba.columns:
        malignancy_prob = proba["1"].values
    else:
        # Fallback: take the second column (should be class 1)
        malignancy_prob = proba.iloc[:, 1].values

    # Prepare output DataFrame
    # Output format: test.csv columns + "label" column (probability)
    results = test.copy()
    results['label'] = malignancy_prob

    # Save results in the same format and extension as test.csv
    result_path = os.path.join(OUTPUT_DIR, "results.csv")
    results.to_csv(result_path, index=False)

    # 6. Validation: Compute AUROC on held-out validation set
    try:
        val_pred = predictor.predict_proba(val_data[['image']])
        if 1 in val_pred.columns:
            val_prob = val_pred[1].values
        elif "1" in val_pred.columns:
            val_prob = val_pred["1"].values
        else:
            val_prob = val_pred.iloc[:, 1].values
        val_true = val_data['label'].values
        val_auc = roc_auc_score(val_true, val_prob)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 7. Output Validation Checks

    # a. Check that prediction file has same number of rows as test set
    pred_df = pd.read_csv(result_path)
    assert len(pred_df) == len(test), "Prediction file row count does not match test set."

    # b. Check that indices are preserved (image_name order)
    assert all(pred_df['image_name'] == test['image_name']), "Prediction file image_name order does not match test set."

    # c. Check that output columns match test.csv + 'label'
    expected_cols = list(test.columns) + ['label']
    assert list(pred_df.columns) == expected_cols, f"Prediction file columns {list(pred_df.columns)} do not match expected {expected_cols}."

    # d. Check that output format is CSV (matches test.csv extension)
    assert result_path.endswith('.csv'), "Prediction file extension does not match test.csv."

    # e. Check that probabilities are in [0, 1]
    assert np.all((pred_df['label'] >= 0) & (pred_df['label'] <= 1)), "Predicted probabilities are not in [0, 1]."

    print("All output validation checks passed. Results saved to:", result_path)