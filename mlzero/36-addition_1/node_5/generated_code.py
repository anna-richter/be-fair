"""
Skin Lesion Malignancy Probability Prediction using AutoGluon MultiModal (Image + Tabular)

This script trains a multimodal model (image + tabular) using AutoGluon MultiModal to predict the probability that a skin lesion is malignant.
It leverages both image and tabular features (including skin_tone for fairness), and uses a strong model architecture (Swin Transformer) with focal loss to address class imbalance.
The script:
- Preprocesses the data (removes NA labels from train, drops index columns, constructs absolute image paths)
- Trains a model and saves it to a timestamped folder in the specified output directory
- Predicts malignancy probabilities for the test set, preserving original indices and output format
- Saves predictions in the same format as the test file, named "results"
- Validates output integrity and prints AUROC on a held-out validation set

# Installation (run in bash before executing this script):
# pip install --upgrade pip
# pip install autogluon.multimodal

Author: AutoML Agent
"""

import os
import time
import uuid
import pandas as pd
import numpy as np
import warnings

from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/36-addition_1/node_5/output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results")

def get_abs_image_path(image_name):
    """Helper to get absolute image path from image_name."""
    return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

def compute_class_weights(labels):
    """Compute class weights for focal loss (malignant=1, benign=0)."""
    labels = np.array(labels, dtype=int)
    counts = np.bincount(labels)
    total = counts.sum()
    # Avoid division by zero
    weights = [float(total) / (2 * float(c)) if c > 0 else 1.0 for c in counts]
    norm = sum(weights)
    weights = [float(w) / float(norm) for w in weights]
    return weights

def fill_missing_test_columns(test_df, train_df):
    """
    Ensure test_df has all columns (except label) present in train_df.
    Fill missing columns with mode (for categorical) or median (for numeric) from train_df.
    """
    for col in train_df.columns:
        if col == "label" or col == "image":
            continue
        if col not in test_df.columns:
            # Use mode for categorical, median for numeric
            if pd.api.types.is_numeric_dtype(train_df[col]):
                fill_value = train_df[col].median()
            else:
                fill_value = train_df[col].mode().iloc[0] if not train_df[col].mode().empty else ""
            test_df[col] = fill_value
    # Ensure column order matches
    test_df = test_df[[c for c in train_df.columns if c != "label"]]
    return test_df

def main():
    # 1. Data Loading and Preprocessing
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples without valid labels (drop NA in 'label')
    train = train.dropna(subset=['label']).reset_index(drop=True)

    # Map label to binary: malignant=1, benign=0
    train['label'] = train['label'].apply(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)
    train['label'] = train['label'].astype(int)

    # Add absolute image path column for AutoGluon
    train['image'] = train['image_name'].apply(get_abs_image_path)
    test['image'] = test['image_name'].apply(get_abs_image_path)

    # 2. Validation Split (10% holdout, stratified)
    train_data, val_data = train_test_split(
        train,
        test_size=0.1,
        random_state=42,
        stratify=train['label']
    )

    # 3. Model Training (Image + Tabular, Swin Transformer, Focal Loss)
    timestamp = int(time.time())
    random_uuid = uuid.uuid4().hex[:8]
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_multimodal_{timestamp}_{random_uuid}")
    os.makedirs(model_dir, exist_ok=True)

    # Compute class weights for focal loss
    class_weights = compute_class_weights(train_data['label'].values)

    # Use both image and tabular features (including skin_tone for fairness)
    # Drop columns that are not useful for training (e.g., expert_opinion is mostly NaN)
    drop_cols = ['expert_opinion']
    train_data_for_fit = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns])
    val_data_for_fit = val_data.drop(columns=[col for col in drop_cols if col in val_data.columns])

    # Model hyperparameters: Swin Transformer, Focal Loss, class weights, more epochs for better performance
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir,
        eval_metric="roc_auc"
    )

    predictor.fit(
        train_data=train_data_for_fit,
        tuning_data=val_data_for_fit,
        hyperparameters={
            "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 20,
        },
        presets="best_quality"
    )

    # 4. Prediction on Test Set
    # Prepare test data for prediction (must have 'image' column and tabular features)
    test_pred_input = test.copy()
    test_pred_input['image'] = test_pred_input['image_name'].apply(get_abs_image_path)
    if 'expert_opinion' in test_pred_input.columns:
        test_pred_input = test_pred_input.drop(columns=['expert_opinion'])

    # Ensure all required columns are present in test_pred_input
    test_pred_input = fill_missing_test_columns(test_pred_input, train_data_for_fit)

    # Predict probabilities (malignancy probability)
    proba = predictor.predict_proba(test_pred_input)
    # For binary, proba is a DataFrame with columns [0, 1], where 1 is the probability of malignant
    if 1 in proba.columns:
        malignancy_prob = proba[1].values
    elif '1' in proba.columns:
        malignancy_prob = proba['1'].values
    else:
        # Fallback: take the second column
        malignancy_prob = proba.iloc[:, 1].values

    # 5. Prepare Output
    output_df = test.copy()
    output_df['label'] = malignancy_prob

    # Ensure output has the same number of rows and indices as test set
    assert len(output_df) == len(test), "Output row count does not match test set!"

    # Save results in the same format as test.csv (CSV)
    allowed_cols = list(test.columns) + ['label']
    output_df = output_df[[col for col in allowed_cols if col in output_df.columns]]

    output_ext = os.path.splitext(TEST_CSV)[1].lower()
    if output_ext == '.csv':
        output_df.to_csv(RESULTS_FILE, index=False)
    elif output_ext in ['.tsv', '.txt']:
        output_df.to_csv(RESULTS_FILE, sep='\t', index=False)
    else:
        raise ValueError(f"Unknown test file extension: {output_ext}")

    # 6. Validation Checks
    pred_df = pd.read_csv(RESULTS_FILE)
    assert len(pred_df) == len(test), "Prediction file row count does not match test set!"
    assert all(pred_df['image_name'] == test['image_name']), "Prediction file image_name order mismatch!"
    expected_cols = list(test.columns) + ['label']
    assert all([col in pred_df.columns for col in expected_cols]), "Prediction file missing required columns!"
    assert os.path.isfile(RESULTS_FILE), "Prediction file not found!"
    assert pred_df['label'].between(0, 1).all(), "Predicted probabilities are not all in [0, 1]!"

    # 7. Validation Metric on Holdout Set
    try:
        val_pred_input = val_data.copy()
        val_pred_input['image'] = val_pred_input['image_name'].apply(get_abs_image_path)
        if 'expert_opinion' in val_pred_input.columns:
            val_pred_input = val_pred_input.drop(columns=['expert_opinion'])
        val_pred_input = fill_missing_test_columns(val_pred_input, train_data_for_fit)
        val_proba = predictor.predict_proba(val_pred_input)
        if 1 in val_proba.columns:
            val_malignancy_prob = val_proba[1].values
        elif '1' in val_proba.columns:
            val_malignancy_prob = val_proba['1'].values
        else:
            val_malignancy_prob = val_proba.iloc[:, 1].values
        val_auc = roc_auc_score(val_data['label'], val_malignancy_prob)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()