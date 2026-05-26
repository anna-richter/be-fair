"""
Skin Lesion Malignancy Prediction Script (AutoGluon MultiModal, Node 3)

This script trains an AutoGluon MultiModal image classifier to predict the probability of malignancy for skin lesion images.
It performs the following steps:
- Loads and preprocesses training and test data (removes NA labels from train, drops index columns, constructs absolute image paths).
- Ensures test data has all columns used in training except the label (required for AutoGluon MultiModal).
- Trains a binary classifier using AutoGluon MultiModal, with fairness consideration for skin tone via focal loss and class weights.
- Saves the trained model to a timestamped folder in the specified output directory.
- Predicts malignancy probabilities (float in [0,1]) for the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file format and column names.
- Validates output integrity (row count, indices, column names, value ranges).
- If possible, evaluates AUROC on a held-out validation set and prints the score.

Installation requirements:
    # pip install --upgrade pip
    # pip install autogluon.multimodal scikit-learn pandas numpy

Usage:
    Place this script in an environment with the required packages and run as main.
    Data must be present in /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data
    Output will be saved to /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/output

Author: AutoML Agent
"""

# Installation steps (uncomment if running in a fresh environment)
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal scikit-learn pandas numpy

import os
import pandas as pd
import numpy as np
import uuid
from datetime import datetime

from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # ------------------- Paths -------------------
    # start change
    # DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
    DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
    # end change
    # start change
    # OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/output"  # original
    OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/39-addition_1"
    # end change
    IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    MODEL_DIR = os.path.join(
        OUTPUT_DIR, f"autogluon_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------- Data Loading -------------------
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train, test]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # Remove training samples with missing labels (do NOT drop from test)
    train = train.dropna(subset=["label"])

    # ------------------- Label Processing -------------------
    # Map label to binary: malignant=1, benign=0
    train["label"] = train["label"].astype(str)
    train["label"] = train["label"].str.strip().str.lower()
    train["label"] = train["label"].map(lambda x: 1 if "malignant" in x else 0)

    # ------------------- Image Path Construction -------------------
    def image_name_to_path(image_name):
        return os.path.abspath(os.path.join(IMAGE_DIR, f"{image_name}.jpg"))

    train["image"] = train["image_name"].apply(image_name_to_path)
    # start change
    # test["image"] = test["image_name"].apply(image_name_to_path)  # original (.jpg)
    test["image"] = test["image_name"].apply(
        lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
    )
    # end change

    # ------------------- Ensure Test Columns Match Training -------------------
    # AutoGluon MultiModal requires all columns used in training (except label) to be present in test
    # Add missing columns to test with placeholder values
    train_cols = set(train.columns) - {"label"}
    for col in train_cols:
        if col not in test.columns:
            # Use np.nan for numeric, "missing" for string columns
            if train[col].dtype.kind in "biufc":
                test[col] = np.nan
            else:
                test[col] = "missing"
    # Ensure column order matches
    test = test[[col for col in train.columns if col != "label"]]

    # ------------------- Fairness: Focal Loss & Class Weights -------------------
    # Compute class weights for focal loss to address imbalance and fairness
    class_counts = train["label"].value_counts().sort_index()
    class_weights = []
    for i in [0, 1]:
        if i in class_counts:
            class_weights.append(1.0 / class_counts[i])
        else:
            class_weights.append(0.0)
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum()  # Normalize

    # ------------------- Validation Split -------------------
    from sklearn.model_selection import train_test_split

    # Hold out 10% for validation (stratified)
    train_data, val_data = train_test_split(
        train,
        test_size=0.1,
        random_state=42,
        stratify=train["label"],
    )

    # ------------------- Model Training -------------------
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=MODEL_DIR,
    )

    # Use focal loss and class weights for fairness
    hyperparameters = {
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights.tolist(),
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        # Optionally, set a reasonable max_epochs for time limit
        # "optim.max_epochs": 20,
    }

    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        # time_limit=1800,  # Uncomment to enforce a time limit (seconds)
    )

    # Save model (already saved in fit, but ensure it's in the correct folder)
    predictor.save(MODEL_DIR)

    # ------------------- Prediction -------------------
    # Prepare test data for prediction (must have all columns except label)
    test_pred_input = test.copy()

    # Predict probabilities (predict_proba returns DataFrame with columns [0,1])
    proba_df = predictor.predict_proba(test_pred_input)
    # For binary, proba_df columns: [0, 1], where 1 is the probability of class 1 (malignant)
    # We want to output the probability of malignancy (class 1)
    malignancy_probs = proba_df[1].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_probs.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # ------------------- Output Formatting -------------------
    # Output file format and extension should match test.csv
    test_ext = os.path.splitext(TEST_CSV)[1]
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")

    # Output column names: match sample submission or test file
    results = test[["image_name"]].copy()
    results["label"] = malignancy_probs

    # Save results in the same format as test.csv (CSV)
    results.to_csv(result_path, index=False)

    # ------------------- Validation Checks -------------------
    # 1. Check row count matches test set
    assert len(results) == len(test), f"Prediction rows ({len(results)}) != test rows ({len(test)})"

    # 2. Check indices preserved (row order matches test set)
    assert all(results["image_name"].values == test["image_name"].values), "Image names do not match test set order"

    # 3. Check output columns
    expected_cols = ["image_name", "label"]
    assert list(results.columns) == expected_cols, f"Output columns {list(results.columns)} != {expected_cols}"

    # 4. Check output format (CSV)
    assert result_path.endswith(".csv"), f"Output file extension {result_path} is not .csv"

    # 5. Check label values are floats in [0,1]
    assert np.all((results["label"] >= 0) & (results["label"] <= 1)), "Predicted probabilities not in [0,1]"

    # 6. Print output file path
    print(f"Predictions saved to: {result_path}")

    # ------------------- Validation Metric (AUROC) -------------------
    try:
        if val_data is not None and len(val_data) > 0:
            val_pred = predictor.predict_proba(val_data)
            val_true = val_data["label"].values
            val_pred_prob = val_pred[1].values
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_true, val_pred_prob)
            print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")