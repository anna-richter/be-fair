"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy (malignant vs. benign)
for skin lesion images, using the AutoGluon MultiModal framework. It performs the following steps:

1. Data preprocessing:
   - Loads train and test CSVs, removes NA labels from train only, and removes unnecessary index columns.
   - Maps image names to absolute image paths for AutoGluon compatibility.
   - Converts multiclass labels to binary (malignant=1, non-neoplastic=0).
   - Ensures test set has all columns used in training except the label.

2. Model training:
   - Trains a MultiModalPredictor on the training set, holding out 10% for validation.
   - Uses focal loss to help with class imbalance.
   - Saves the trained model to a timestamped folder in the output directory.

3. Prediction:
   - Predicts malignancy probability (float in [0,1]) for each test image, preserving original indices.
   - Saves results in the same format and extension as the test file, with correct column names.

4. Validation:
   - Computes AUROC on the held-out validation set and prints the score.
   - Performs strict validation checks on the output file.

5. Documentation and reproducibility:
   - Includes installation steps and comments for clarity.
   - All outputs are saved to the specified output directory.

Requirements:
- Python 3.8+
- AutoGluon >= 0.8.2

Usage:
- Place this script in any location and run it. It expects the data folder as described in the task.
"""

# Installation steps (uncomment if running in a fresh environment)
# !pip install --upgrade pip
# !pip install autogluon.multimodal

import os
import sys
import uuid
import time
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # ------------------- CONFIGURATION -------------------
    # start change
    # DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
    DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
    # end change
    # start change
    # OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/41-addition_1/node_4/output"  # original
    OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/41-addition_1"
    # end change
    IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")

    # ------------------- DATA LOADING -------------------
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples without valid labels (drop NA from train only)
    train = train.dropna(subset=['label'])

    # ------------------- LABEL PROCESSING -------------------
    # Map label to binary: malignant=1, non-neoplastic=0
    def label_to_binary(label):
        if str(label).strip().lower() == "malignant":
            return 1
        else:
            return 0

    train['label'] = train['label'].apply(label_to_binary)

    # ------------------- IMAGE PATHS -------------------
    # Add absolute image path column for AutoGluon
    def image_name_to_path(image_name):
        return os.path.abspath(os.path.join(IMAGE_DIR, f"{image_name}.jpg"))

    train['image'] = train['image_name'].apply(image_name_to_path)
    # start change
    # test['image'] = test['image_name'].apply(image_name_to_path)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
    )
    # end change

    # ------------------- ENSURE TEST COLUMNS MATCH TRAIN -------------------
    # AutoGluon MultiModal requires all columns used in training (except label) to be present in test
    # Add missing columns to test with default values (np.nan)
    train_cols = set(train.columns) - {'label'}
    for col in train_cols:
        if col not in test.columns:
            test[col] = np.nan
    # Ensure column order matches
    test = test[[col for col in train.columns if col != 'label']]

    # ------------------- FAIRNESS: SKIN TONE -------------------
    # We include 'skin_tone' as a feature to encourage fairness, but do not use it as a label.
    # AutoGluon will treat it as a tabular feature.

    # ------------------- TRAIN/VALIDATION SPLIT -------------------
    # Hold out 10% of training data for validation
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(train, train['label']))
    train_data = train.iloc[train_idx].reset_index(drop=True)
    val_data = train.iloc[val_idx].reset_index(drop=True)

    # ------------------- CLASS IMBALANCE HANDLING -------------------
    # Compute class weights for focal loss
    class_counts = train_data['label'].value_counts().sort_index()
    num_classes = 2
    class_weights = []
    total = class_counts.sum()
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        class_weights.append(total / (num_classes * count))
    class_weights = np.array(class_weights) / np.sum(class_weights)

    # ------------------- MODEL TRAINING -------------------
    from autogluon.multimodal import MultiModalPredictor

    # Create a unique model directory with random timestamp
    model_dir = os.path.join(
        OUTPUT_DIR,
        f"autogluon_model_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(model_dir, exist_ok=True)

    # Train the model
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir
    )

    # Use focal loss for class imbalance, and include skin_tone as a feature
    predictor.fit(
        train_data=train_data,
        time_limit=1800,  # 30 minutes max for training
        hyperparameters={
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights.tolist(),
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 20,
        },
    )

    # ------------------- PREDICTION -------------------
    # Predict probabilities for the test set (must preserve original indices)
    # AutoGluon expects the same columns as in training except for the label
    test_for_pred = test.copy()

    # Predict_proba returns a DataFrame with columns [0, 1] for binary classification
    proba = predictor.predict_proba(test_for_pred)
    # The probability of class 1 ("malignant") is in column 1
    if 1 in proba.columns:
        malignancy_prob = proba[1]
    elif "1" in proba.columns:
        malignancy_prob = proba["1"]
    else:
        # Fallback: take the last column
        malignancy_prob = proba.iloc[:, -1]

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_prob.astype(float).values,
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare output DataFrame
    output_df = test[['image_name']].copy()
    output_df['label'] = malignancy_prob.values  # Column name must match training file

    # Save results in the same format and extension as test.csv
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    if test_ext == ".csv":
        output_df.to_csv(result_path, index=False)
    elif test_ext == ".parquet":
        output_df.to_parquet(result_path, index=False)
    else:
        # Default to CSV if unknown
        output_df.to_csv(result_path, index=False)

    # ------------------- VALIDATION -------------------
    # Compute AUROC on held-out validation set
    try:
        val_for_pred = val_data.copy()
        # Remove label column for prediction
        val_pred_input = val_for_pred.drop(columns=['label'])
        # Ensure all columns used in training (except label) are present in val_pred_input
        for col in train_cols:
            if col not in val_pred_input.columns:
                val_pred_input[col] = np.nan
        val_pred_input = val_pred_input[[col for col in train.columns if col != 'label']]
        val_proba = predictor.predict_proba(val_pred_input)
        if 1 in val_proba.columns:
            val_malignancy_prob = val_proba[1]
        elif "1" in val_proba.columns:
            val_malignancy_prob = val_proba["1"]
        else:
            val_malignancy_prob = val_proba.iloc[:, -1]
        val_true = val_data['label'].values
        val_score = roc_auc_score(val_true, val_malignancy_prob)
        print(f"Validation AUROC: {val_score:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # ------------------- OUTPUT VALIDATION CHECKS -------------------
    # 1. Check that the output file has the same number of rows as the test set
    output_df_check = pd.read_csv(result_path) if test_ext == ".csv" else pd.read_parquet(result_path)
    assert len(output_df_check) == len(test), "Number of predictions does not match number of test samples."

    # 2. Check that the indices are preserved (row order matches test set)
    assert all(output_df_check['image_name'].values == test['image_name'].values), \
        "Output image_name order does not match test set."

    # 3. Check that the output columns match requirements
    assert list(output_df_check.columns) == ['image_name', 'label'], \
        f"Output columns {list(output_df_check.columns)} do not match required ['image_name', 'label']"

    # 4. Check that all predictions are floats in [0, 1]
    assert np.all((output_df_check['label'] >= 0) & (output_df_check['label'] <= 1)), \
        "Predicted probabilities are not all in [0, 1]"

    # 5. Check output format
    if test_ext == ".csv":
        assert result_path.endswith(".csv"), "Output file extension mismatch."
    elif test_ext == ".parquet":
        assert result_path.endswith(".parquet"), "Output file extension mismatch."

    print(f"Prediction results saved to: {result_path}")
    print(f"Trained model saved to: {model_dir}")