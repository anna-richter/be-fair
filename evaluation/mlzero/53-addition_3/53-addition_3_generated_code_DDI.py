"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy for skin lesion images,
with a focus on fairness across skin tones. It uses AutoGluon MultiModal for binary classification, applies
class balancing to mitigate skin tone bias, and outputs malignancy probabilities for the test set.

- Input: train.csv, test.csv, and JPEG images in MyImages/
- Output: malignancy probabilities for each test image, saved in the same format as test.csv, with original indices preserved.
- Model and results are saved to: /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/53-addition_3/node_18/output

Installation requirements (run before execution if needed):
    # pip install --upgrade pip
    # pip install autogluon.multimodal pandas scikit-learn

Usage:
    Place this script in the working directory and run with Python 3.7+.
    Ensure all data files are present at the specified absolute paths.

Author: AutoML Agent
"""

import os
import time
import uuid
import random
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==== Installation steps (uncomment if needed) ====
# !pip install --upgrade pip
# !pip install autogluon.multimodal pandas scikit-learn

# ==== Paths ====
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/53-addition_3/node_18/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/53-addition_3"
# end change
IMAGES_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

def get_absolute_image_path(image_name):
    """Return absolute path to image file given image_name (without extension)."""
    return os.path.abspath(os.path.join(IMAGES_DIR, f"{image_name}.jpg"))

def map_label_to_binary(label):
    """Map string label to binary: malignant=1, non-neoplastic=0."""
    return 1 if str(label).strip().lower() == "malignant" else 0

def compute_class_weights(df, label_col):
    """
    Compute class weights for focal loss: inverse of class frequency, normalized.
    Returns a list of weights in order [class_0, class_1]
    """
    class_counts = df[label_col].value_counts().sort_index().to_dict()
    total = sum(class_counts.values())
    weights = [total / (len(class_counts) * class_counts.get(i, 1)) for i in range(2)]
    weights = np.array(weights) / np.sum(weights)
    return weights.tolist()

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def get_random_model_dir(base_dir):
    """Generate a random timestamped model directory under base_dir."""
    ts = int(time.time())
    rand = uuid.uuid4().hex[:8]
    model_dir = os.path.join(base_dir, f"automm_model_{ts}_{rand}")
    return model_dir

def validate_output(pred_df, test_df, label_col, output_path):
    """Validation checks for output predictions."""
    # 1. Indices must match
    assert (pred_df.index == test_df.index).all(), "Prediction indices do not match test indices"
    # 2. Column names
    assert label_col in pred_df.columns, f"Output missing required column: {label_col}"
    # 3. Output format
    assert pred_df.shape[0] == test_df.shape[0], "Number of predictions does not match test set"
    # 4. Output file format
    ext = os.path.splitext(output_path)[1]
    if ext == ".csv":
        pd.read_csv(output_path)
    # 5. Sanity check: probabilities in [0,1]
    assert pred_df[label_col].between(0, 1).all(), "Predicted probabilities not in [0,1]"

if __name__ == "__main__":
    # ==== 1. Data Loading and Preprocessing ====
    # Load train and test data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train_df, test_df]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # Drop NA labels from training data only
    train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)

    # Map labels to binary (malignant=1, non-neoplastic=0)
    train_df["label"] = train_df["label"].map(map_label_to_binary)

    # Add absolute image path column for AutoGluon
    train_df["image"] = train_df["image_name"].apply(get_absolute_image_path)
    # start change
    # test_df["image"] = test_df["image_name"].apply(get_absolute_image_path)  # original (.jpg)
    test_df["image"] = test_df["image_name"].apply(
        lambda x: os.path.abspath(os.path.join(IMAGES_DIR, f"{x}.png"))
    )
    # end change

    # ==== 2. Class Balancing ====
    label_col = "label"
    class_weights = compute_class_weights(train_df, label_col)

    # ==== 3. Train/Validation Split ====
    # Hold out 10% for validation (stratified by label)
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.10,
        random_state=SEED,
        stratify=train_df[label_col]
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # ==== 4. Model Training ====
    from autogluon.multimodal import MultiModalPredictor

    # Model directory
    model_dir = get_random_model_dir(OUTPUT_DIR)
    ensure_dir(model_dir)

    # Prepare hyperparameters for class imbalance and improved architecture
    hyperparameters = {
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights,
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        # Use a stronger backbone for better performance
        "model.timm_image.checkpoint_name": "convnext_base_in22ft1k",
        "env.num_gpus": 1,
        "optim.patience": 5,
        "env.precision": "16-mixed",
        # Optionally, increase number of epochs for better convergence
        "optim.max_epochs": 20,
    }

    # Train predictor
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_dir
    )

    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters=hyperparameters,
        time_limit=3600 - 300,  # Leave 5 min for inference and validation
    )

    # ==== 5. Prediction on Test Set ====
    # Prepare test data for prediction (must have 'image' column)
    test_pred_df = test_df.copy()
    # Ensure all columns present in train except label are present in test (fill missing with np.nan)
    required_cols = [col for col in train_df.columns if col not in ["label"]]
    for col in required_cols:
        if col not in test_pred_df.columns:
            test_pred_df[col] = np.nan
    # Ensure column order matches
    test_pred_df = test_pred_df[[col for col in train_df.columns if col != "label"]]

    # Predict_proba returns a DataFrame with columns [0, 1] for binary
    proba = predictor.predict_proba(test_pred_df)
    # We want the probability of malignant (class 1)
    if 1 in proba.columns:
        test_pred_df[label_col] = proba[1]
    else:
        test_pred_df[label_col] = proba["1"]

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test_df["image_name"].astype(str) + ".png",
        "predicted_probability": test_pred_df["label"].astype(float).values,
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Output DataFrame: must have original indices and required columns
    output_df = test_pred_df[[label_col]].copy()
    output_df.index = test_df.index  # Ensure indices match

    # ==== 6. Save Results ====
    # Save with same format/extension as test.csv
    ensure_dir(OUTPUT_DIR)
    result_path = os.path.join(OUTPUT_DIR, "results.csv")
    output_df.to_csv(result_path, index=True)

    # ==== 7. Validation Checks ====
    validate_output(output_df, test_df, label_col, result_path)

    # ==== 8. Validation on Hold-out Set ====
    try:
        val_pred_proba = predictor.predict_proba(val_data)
        if 1 in val_pred_proba.columns:
            val_probs = val_pred_proba[1].values
        else:
            val_probs = val_pred_proba["1"].values
        val_labels = val_data[label_col].values
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"Validation ROC-AUC: {val_auc:.4f}")

        # Fairness: AUROC gap between light and dark skin tones
        group_col = "skin_tone"
        val_skin_tone = val_data[group_col].values
        light_mask = np.isin(val_skin_tone, [1, 2])
        dark_mask = np.isin(val_skin_tone, [3, 4])
        if np.any(light_mask) and np.any(dark_mask):
            light_auc = roc_auc_score(val_labels[light_mask], val_probs[light_mask])
            dark_auc = roc_auc_score(val_labels[dark_mask], val_probs[dark_mask])
            gap = abs(light_auc - dark_auc)
            print(f"Validation AUROC (light skin): {light_auc:.4f}")
            print(f"Validation AUROC (dark skin): {dark_auc:.4f}")
            print(f"Validation AUROC gap (light vs dark): {gap:.4f}")
        else:
            print("Not enough samples for both light and dark skin tones in validation set.")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Results saved to: {result_path}")
    print(f"Model saved to: {model_dir}")