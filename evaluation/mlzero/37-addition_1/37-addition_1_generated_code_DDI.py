"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image+tabular classification model to predict the probability of malignancy for skin lesions
from RGB JPEG images, using the AutoGluon MultiModalPredictor. It performs all required data preprocessing,
model training (with fairness consideration for skin tone), prediction, and validation checks. The script
saves the trained model and prediction results in the specified output directory, ensuring strict adherence
to input/output formats and index preservation.

Installation requirements (run before executing this script if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Ensure the data folder and output folder are accessible as described in the task.
    Run this script as a standalone module.

Author: AutoML Agent
"""

import os
import time
import random
import warnings

import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Paths
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/37-addition_1/node_5/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/37-addition_1"
# end change
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "MyImages")

# Output file name (must match test file extension)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_image_path(image_name):
    """Helper to get absolute path for an image file given its name (without extension)."""
    return os.path.abspath(os.path.join(IMAGES_DIR, f"{image_name}.jpg"))

def main():
    # 1. Data Loading and Preprocessing
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

    # Drop training samples without valid labels (NA in 'label')
    train = train.dropna(subset=['label'])

    # Map label to binary: malignant=1, benign=0
    train['label'] = train['label'].apply(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)

    # Add absolute image path column for AutoGluon
    train['image'] = train['image_name'].apply(get_image_path)
    # start change
    # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMAGES_DIR, f"{x}.png"))
    )
    # end change

    # Ensure all tabular columns used in training are present in test, fill missing with default values
    tabular_cols = ['skin_tone', 'alternative_skin_tone', 'expert_opinion']
    for col in tabular_cols:
        if col not in test.columns:
            # Use -1 for numeric/categorical, np.nan for text
            if col in ['skin_tone', 'alternative_skin_tone']:
                test[col] = -1
            else:
                test[col] = np.nan

    # 3. Validation Split (10% holdout)
    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_frac = 0.1
    val_size = int(len(train) * val_frac)
    val_data = train.iloc[:val_size].reset_index(drop=True)
    train_data = train.iloc[val_size:].reset_index(drop=True)

    # 4. Model Training
    timestamp = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Use focal loss for class imbalance (malignant is likely minority)
    class_counts = train_data['label'].value_counts().sort_index()
    if 0 not in class_counts:
        class_counts[0] = 0
    if 1 not in class_counts:
        class_counts[1] = 0
    total = class_counts.sum()
    weights = [1.0 / (class_counts[0] / total + 1e-8), 1.0 / (class_counts[1] / total + 1e-8)]
    weights = (np.array(weights) / np.sum(weights)).tolist()

    predictor = MultiModalPredictor(
        label="label",
        problem_type="binary",
        path=model_dir,
        eval_metric="roc_auc"
    )

    predictor.fit(
        train_data=train_data,
        time_limit=None,
        hyperparameters={
            "model.timm_image.checkpoint_name": "convnext_tiny",
            "env.num_gpus": 1,
            "env.num_workers": 8,
            "env.precision": "16-mixed",
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": weights,
            "optim.focal_loss.gamma": 1.0,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 20,
            "optim.patience": 7,
            "optim.val_check_interval": 0.25,
            "optim.gradient_clip_val": 2,
            "optim.gradient_clip_algorithm": "norm",
            "optim.top_k": 5,
            "optim.top_k_average_method": "uniform_soup",
        },
        seed=SEED,
    )

    predictor.save(model_dir)

    # 5. Prediction on Test Set
    # Ensure test set indices are preserved and all columns match training columns except label
    test_pred_input = test.copy()
    input_cols = ['image', 'skin_tone', 'alternative_skin_tone', 'expert_opinion']
    test_pred_input = test_pred_input[input_cols]

    proba = predictor.predict_proba(test_pred_input)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            malignancy_proba = proba[1].values
        else:
            malignancy_proba = proba[str(1)].values
    else:
        malignancy_proba = proba[:, 1]

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_proba.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    results = test[['image_name']].copy()
    results['label'] = malignancy_proba
    results.index = test.index

    results.to_csv(RESULTS_FILE, index=False)

    # 6. Validation Checks
    assert os.path.exists(RESULTS_FILE), f"Results file not found: {RESULTS_FILE}"
    results_check = pd.read_csv(RESULTS_FILE)
    assert len(results_check) == len(test), f"Number of predictions ({len(results_check)}) does not match test set ({len(test)})"
    expected_cols = ['image_name', 'label']
    assert list(results_check.columns) == expected_cols, f"Output columns {list(results_check.columns)} do not match expected {expected_cols}"
    assert all(results_check['image_name'] == test['image_name']), "Image names in results do not match test set order"
    assert np.all((results_check['label'] >= 0) & (results_check['label'] <= 1)), "Predicted probabilities are not in [0, 1]"

    # 7. Validation Metric on Holdout Set
    try:
        val_pred_input = val_data[input_cols]
        val_true = val_data['label'].values
        val_proba = predictor.predict_proba(val_pred_input)
        if isinstance(val_proba, pd.DataFrame):
            if 1 in val_proba.columns:
                val_pred = val_proba[1].values
            else:
                val_pred = val_proba[str(1)].values
        else:
            val_pred = val_proba[:, 1]
        val_auc = roc_auc_score(val_true, val_pred)
        print(f"Validation AUROC: {val_auc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()