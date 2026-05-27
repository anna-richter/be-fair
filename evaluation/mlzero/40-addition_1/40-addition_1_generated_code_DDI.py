"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script:
- Loads and preprocesses skin lesion image data for binary classification (malignant vs. benign).
- Trains an AutoGluon MultiModalPredictor on the training data, using image files as input and skin tone features for fairness.
- Handles class imbalance using focal loss with computed class weights.
- Evaluates the model using AUROC on a held-out validation set.
- Predicts malignancy probabilities (float in [0,1]) for the test set, preserving original indices and output format.
- Saves the trained model and prediction results to the specified output directory.
- Performs validation checks to ensure output correctness.

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Place this script in any location and run it. It expects the following data structure:
        /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/
            - train.csv
            - test.csv
            - MyImages/*.jpg

Output:
    - Trained model folder (with random timestamp) in:
        /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/40-addition_1/node_10/output/
    - Prediction file "results" in the same format as test.csv, with malignancy probabilities.
    - Validation AUROC printed to stdout.

Note:
    - The script is wrapped in if __name__ == "__main__" to avoid DDP errors.
    - Errors are not silently caught except for validation, which is allowed to fail gracefully.
"""

import os
import uuid
import time
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ======================== PATHS ========================
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/40-addition_1/node_10/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/40-addition_1"
# end change

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_model_path():
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().hex[:8]
    return os.path.join(OUTPUT_DIR, f"autogluon_model_{ts}_{rand}")

def compute_class_weights(labels):
    bincount = np.bincount(labels)
    total = np.sum(bincount)
    weights = [total / (2 * c) if c > 0 else 0.0 for c in bincount]
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights.tolist()

def map_label_to_binary(label):
    if isinstance(label, str):
        label = label.strip().lower()
        if label in ["malignant", "malign", "cancer", "1"]:
            return 1
        elif label in ["benign", "non-neoplastic", "0"]:
            return 0
    elif isinstance(label, (int, float)):
        return int(label)
    return np.nan

if __name__ == "__main__":
    # ======================== DATA LOADING ========================
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # Remove unnecessary index column if present
    for col in ['Unnamed: 0', 'index']:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    # Drop training samples without valid labels (do NOT drop from test)
    train = train.dropna(subset=['label'])

    # Map labels to binary (malignant=1, benign=0)
    train['label'] = train['label'].apply(map_label_to_binary)
    train = train.dropna(subset=['label'])
    train['label'] = train['label'].astype(int)

    # ======================== IMAGE PATHS ========================
    train['image'] = train['image_name'].apply(lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.jpg")))
    # start change
    # test['image'] = test['image_name'].apply(lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.jpg")))  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
    )
    # end change

    # ======================== FAIRNESS: SKIN TONE ========================
    fairness_features = []
    for col in ['skin_tone', 'alternative_skin_tone']:
        if col in train.columns:
            fairness_features.append(col)
            # If missing in test, add with default -1
            if col not in test.columns:
                test[col] = -1
            train[col] = train[col].fillna(-1)
            test[col] = test[col].fillna(-1)

    # ======================== TRAIN/VALIDATION SPLIT ========================
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train,
        test_size=0.1,
        stratify=train['label'],
        random_state=42,
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # ======================== CLASS WEIGHTS FOR FOCAL LOSS ========================
    class_weights = compute_class_weights(train_data['label'].values)

    # ======================== MODEL TRAINING ========================
    model_path = get_random_model_path()
    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_path,
    )

    # Only keep columns needed: image, label, fairness features
    train_cols = ['image', 'label'] + fairness_features
    train_data_ag = train_data[train_cols].copy()
    val_data_ag = val_data[train_cols].copy()

    # Set a time limit to avoid wall-time kill (e.g., 3300 seconds)
    time_limit = 3300

    # Fit model (remove unsupported eval_metric and val_data arguments)
    predictor.fit(
        train_data=train_data_ag,
        hyperparameters={
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": class_weights,
            "optim.focal_loss.gamma": 2.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 12,  # Lowered for time safety
        },
        holdout_frac=0.1,  # Use 10% for validation
        presets="medium_quality",  # Lowered for time safety
        seed=42,
        time_limit=time_limit
    )

    # ======================== VALIDATION EVALUATION ========================
    try:
        val_pred_proba = predictor.predict_proba(val_data_ag)
        # val_pred_proba is a DataFrame with columns [0, 1] (benign, malignant)
        # We want probability of malignant (class 1)
        if 1 in val_pred_proba.columns:
            val_malignant_proba = val_pred_proba[1].values
        else:
            val_malignant_proba = val_pred_proba[str(1)].values
        val_true = val_data_ag['label'].values
        val_auc = roc_auc_score(val_true, val_malignant_proba)
        print(f"Validation AUROC: {val_auc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # ======================== TEST PREDICTION ========================
    # Prepare test data for prediction (must include all fairness features if present)
    test_cols = ['image'] + fairness_features
    test_data_ag = test[test_cols].copy()

    # Predict probabilities for test set
    test_pred_proba = predictor.predict_proba(test_data_ag)
    if 1 in test_pred_proba.columns:
        test_malignant_proba = test_pred_proba[1].values
    else:
        test_malignant_proba = test_pred_proba[str(1)].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": test_malignant_proba.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # ======================== OUTPUT FORMATTING ========================
    output_df = test.copy()
    output_df['label'] = test_malignant_proba

    # Ensure output has the same number of rows and indices as test set
    assert len(output_df) == len(test), "Output row count does not match test set"
    assert (output_df.index == test.index).all(), "Output indices do not match test set"

    # Save results in the same format/extension as test.csv (CSV)
    result_path = os.path.join(OUTPUT_DIR, "results")
    ext = os.path.splitext(TEST_CSV)[1].lower()
    if ext == ".csv":
        output_df.to_csv(result_path, index=False)
    elif ext in [".tsv", ".txt"]:
        output_df.to_csv(result_path, sep='\t', index=False)
    else:
        output_df.to_csv(result_path, index=False)

    # ======================== VALIDATION CHECKS ========================
    assert os.path.exists(result_path), "Result file was not created"
    out_df = pd.read_csv(result_path)
    expected_cols = list(test.columns) + ['label']
    assert list(out_df.columns) == expected_cols, f"Output columns {list(out_df.columns)} do not match expected {expected_cols}"
    assert len(out_df) == len(test), "Output row count does not match test set"
    assert np.all((out_df['label'] >= 0) & (out_df['label'] <= 1)), "Predicted probabilities are not in [0,1]"
    if 'image_name' in out_df.columns and 'image_name' in test.columns:
        assert (out_df['image_name'] == test['image_name']).all(), "image_name column does not match test set"

    print(f"Prediction results saved to: {result_path}")
    print(f"Trained model saved to: {model_path}")