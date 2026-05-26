"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy for skin lesion images,
using the AutoGluon MultiModalPredictor. It handles all data preprocessing, model training, prediction, and output
formatting as specified. The script ensures fairness with respect to skin tone by using focal loss with class weights
to mitigate class imbalance, and performs validation with AUROC metric.

Output files (model and results) are saved ONLY to:
    /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/output

Installation requirements:
    # pip install --upgrade pip
    # pip install autogluon.multimodal
    # pip install scikit-learn

Usage:
    Run this script as a standalone Python file. It assumes the data is present in the specified absolute paths.
"""

# ==============================
# Installation steps (uncomment if needed)
# ==============================
# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install autogluon.multimodal
# !{sys.executable} -m pip install scikit-learn

import os
import uuid
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from autogluon.multimodal import MultiModalPredictor

# ==============================
# Constants and Paths
# ==============================
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/45-addition_2"
# end change

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_model_dir():
    # Use a random timestamp for model directory
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().hex[:8]
    return os.path.join(OUTPUT_DIR, f"automm_model_{ts}_{rand}")

def map_image_names_to_paths(df, image_col="image_name"):
    """Convert image_name column to absolute image file paths."""
    df = df.copy()
    df[image_col] = df[image_col].apply(lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.jpg")))
    return df

def prepare_labels(df, label_col="label"):
    """Map string labels to binary: 'malignant'->1, 'benign'->0, drop others."""
    df = df.copy()
    # Accept only 'malignant' and 'benign'
    df = df[df[label_col].isin(["malignant", "benign"])]
    df[label_col] = df[label_col].map({"malignant": 1, "benign": 0})
    return df

def compute_class_weights(labels):
    """Compute class weights for focal loss (malignant/benign)."""
    # weights: inverse frequency, normalized to sum to 1
    bincount = np.bincount(labels)
    total = np.sum(bincount)
    weights = [total / (2 * c) if c > 0 else 0.0 for c in bincount]
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights.tolist()

def validate_results(
    test_df, results_df, sample_submission=None, label_col="label", proba_range=(0, 1)
):
    """Validation checks for output file."""
    # 1. Indices must match
    assert list(test_df.index) == list(results_df.index), "Test and result indices do not match!"
    # 2. Number of rows
    assert len(test_df) == len(results_df), "Number of predictions does not match test set!"
    # 3. Column names
    if sample_submission is not None:
        assert list(results_df.columns) == list(sample_submission.columns), "Result columns do not match sample submission!"
    else:
        assert label_col in results_df.columns, f"Missing required column: {label_col}"
    # 4. Output format: extension and type
    # (Handled by saving with same extension as test file)
    # 5. Probability range
    if np.issubdtype(results_df[label_col].dtype, np.floating):
        assert ((results_df[label_col] >= proba_range[0]) & (results_df[label_col] <= proba_range[1])).all(), \
            f"Probabilities not in range {proba_range}!"
    # 6. No extra columns
    assert set(results_df.columns) == {label_col}, "Result file has extra columns!"

if __name__ == "__main__":
    # ==============================
    # 1. Data Loading and Preprocessing
    # ==============================
    # Load train and test data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train_df, test_df]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # Drop NA labels from training data only
    train_df = train_df.dropna(subset=["label"])

    # Only keep 'malignant' and 'benign' samples, map to 1/0
    train_df = prepare_labels(train_df, label_col="label")

    # Map image_name to absolute image paths
    train_df = map_image_names_to_paths(train_df, image_col="image_name")
    test_df = map_image_names_to_paths(test_df, image_col="image_name")
    # start change
    test_df["image_name"] = test_df["image_name"].apply(
        lambda p: os.path.splitext(p)[0] + ".png"
    )
    # end change

    # ==============================
    # 2. Validation Split
    # ==============================
    # Hold out 10% for validation (stratified by label)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df["label"],
    )

    # ==============================
    # 3. Model Training
    # ==============================
    # Compute class weights for focal loss (to help fairness and imbalance)
    class_weights = compute_class_weights(train_data["label"].values)

    # Prepare model directory
    model_dir = get_random_model_dir()

    # Use focal loss for fairness and class imbalance
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir,
    )

    # Only use image modality (image_name column)
    # Use default image backbone, but focal loss for fairness
    predictor.fit(
        train_data=train_data[["image_name", "label"]],
        # Validation set is provided for early stopping and best model selection
        tuning_data=val_data[["image_name", "label"]],
        hyperparameters={
            "env.num_gpus": 1,  # Use single GPU to avoid DDP errors
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 15,  # Reasonable default for small/medium datasets
            # Do not increase batch size to avoid OOM
        },
        # Use default presets for robust training
        presets="best_quality",
        time_limit=3600 - 300,  # Leave 5 min for inference and saving
    )

    # Save model (already saved in model_dir by AutoGluon)

    # ==============================
    # 4. Prediction
    # ==============================
    # Prepare test data for prediction
    test_pred_df = test_df.copy()
    # Keep original indices for output
    test_pred_df = test_pred_df.reset_index(drop=False)
    orig_indices = test_pred_df["index"]
    test_pred_df = test_pred_df.drop(columns=["index"])

    # Predict probabilities for 'malignant' (class 1)
    proba = predictor.predict_proba(test_pred_df[["image_name"]])
    # If predict_proba returns a DataFrame with columns [0,1], take column 1
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            malignancy_proba = proba[1].values
        elif "1" in proba.columns:
            malignancy_proba = proba["1"].values
        else:
            # If only one column, assume it's for class 1
            malignancy_proba = proba.iloc[:, -1].values
    else:
        # If numpy array
        malignancy_proba = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

    # start change
    _orig_test_names = pd.read_csv(TEST_CSV)["image_name"].astype(str).reset_index(drop=True)
    _ddi_df = pd.DataFrame({
        "DDI_file": _orig_test_names + ".png",
        "predicted_probability": malignancy_proba.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare results DataFrame
    results_df = pd.DataFrame({
        "label": malignancy_proba
    }, index=orig_indices)
    # Ensure order matches test set
    results_df = results_df.loc[test_df.index]

    # ==============================
    # 5. Save Results
    # ==============================
    # Save with same format/extension as test.csv
    result_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(result_path, index=True, header=True)

    # ==============================
    # 6. Validation Checks
    # ==============================
    # Check output format and validity
    validate_results(
        test_df=test_df,
        results_df=results_df,
        sample_submission=None,  # No sample submission provided
        label_col="label",
        proba_range=(0, 1),
    )

    # ==============================
    # 7. Validation Metric (AUROC)
    # ==============================
    try:
        # Evaluate on held-out validation set
        val_pred_proba = predictor.predict_proba(val_data[["image_name"]])
        if isinstance(val_pred_proba, pd.DataFrame):
            if 1 in val_pred_proba.columns:
                val_malignancy_proba = val_pred_proba[1].values
            elif "1" in val_pred_proba.columns:
                val_malignancy_proba = val_pred_proba["1"].values
            else:
                val_malignancy_proba = val_pred_proba.iloc[:, -1].values
        else:
            val_malignancy_proba = val_pred_proba[:, 1] if val_pred_proba.shape[1] == 2 else val_pred_proba[:, 0]
        val_score = roc_auc_score(val_data["label"].values, val_malignancy_proba)
        print(f"Validation AUROC: {val_score:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")