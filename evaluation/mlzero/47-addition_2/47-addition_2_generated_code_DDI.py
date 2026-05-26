"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script trains an image classification model to predict the probability of malignancy
for skin lesion images, using AutoGluon MultiModal. It:
- Loads and preprocesses the data (removes NA labels, drops index columns, filters to 'benign' and 'malignant')
- Trains a model (with class balancing for fairness)
- Predicts malignancy probabilities for the test set, preserving original indices and format
- Saves predictions to the required output directory, matching the test file format
- Performs validation (AUROC) on a held-out validation set, and checks output integrity

Installation requirements (run before execution if needed):
    # pip install --upgrade pip
    # pip install autogluon.multimodal scikit-learn pandas

Author: AutoML Agent
"""

import os
import uuid
import time
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set constants for data and output paths
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/47-addition_2/node_16/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/47-addition_2"
# end change

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_path(image_name):
    """Helper to get absolute image path from image_name (without extension)"""
    return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

def compute_class_weights(labels):
    """Compute class weights for focal loss (for fairness and imbalance)"""
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = []
    for c in classes:
        weights.append(1.0 / (np.sum(labels == c) / total))
    weights = np.array(weights)
    weights = weights / weights.sum()
    return dict(zip(classes, weights))

if __name__ == "__main__":
    # 1. Load data
    train = pd.read_csv(TRAIN_CSV)
    # start change
    # test = pd.read_csv(TEST_CSV)
    test = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # 2. Data preprocessing
    # Remove unnecessary index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remove training samples with missing labels (do NOT drop from test)
    train = train.dropna(subset=['label'])

    # Only keep necessary columns
    train = train[['image_name', 'label', 'skin_tone']]
    test = test[['image_name']]

    # Filter to only 'benign' and 'malignant' labels (exclude 'non-neoplastic')
    train = train[train['label'].isin(['benign', 'malignant'])].reset_index(drop=True)

    # Map image_name to absolute image path
    train['image'] = train['image_name'].apply(get_image_path)
    # start change
    # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
    )
    # end change

    # For AutoGluon, label must be string or int; ensure it's string
    train['label'] = train['label'].astype(str)

    # Let's check unique labels
    unique_labels = sorted(train['label'].unique())
    if set(unique_labels) == set(['benign', 'malignant']):
        label_order = ['benign', 'malignant']
    else:
        label_order = unique_labels
        print(f"Warning: Unexpected label set: {unique_labels}")

    # 3. Validation split (10% holdout, stratified)
    train_data, val_data = train_test_split(
        train,
        test_size=0.10,
        random_state=42,
        stratify=train['label']
    )

    # 4. Compute class weights for focal loss (for fairness)
    class_weights = compute_class_weights(train_data['label'].values)
    # Order weights according to label_order and ensure float type
    class_weights_list = [float(class_weights[lab]) for lab in label_order]

    # 5. Model training
    from autogluon.multimodal import MultiModalPredictor

    # Use a random timestamp for model folder
    model_folder = os.path.join(
        OUTPUT_DIR,
        f"autogluon_model_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(model_folder, exist_ok=True)

    # Use focal loss for class imbalance/fairness
    predictor = MultiModalPredictor(
        label="label",
        problem_type="classification",
        path=model_folder
    )

    # Use only 1 GPU (to avoid DDP errors)
    hyperparameters = {
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights_list,
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        "optim.max_epochs": 10,
        # Optionally, use a strong image backbone (commented for OOM safety)
        # "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    }

    # Fit model
    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters=hyperparameters,
        presets="best_quality",
        time_limit=3600-600,  # Leave time for inference
    )

    # 6. Prediction on test set
    # Prepare test data for prediction (must have 'image' and 'skin_tone' columns)
    # Fill skin_tone in test with most common value from train if missing
    if 'skin_tone' not in test.columns:
        most_common_skin_tone = train['skin_tone'].mode().iloc[0]
        test['skin_tone'] = most_common_skin_tone
    # Ensure column order matches training (excluding label)
    test_pred_df = test[['image_name', 'skin_tone', 'image']]

    # Predict_proba returns a DataFrame with columns as class names
    proba_df = predictor.predict_proba(test_pred_df)

    # The output should be a single column: probability of 'malignant'
    if 'malignant' in proba_df.columns:
        malignancy_proba = proba_df['malignant']
    else:
        malignancy_proba = proba_df.iloc[:, -1]
        print("Warning: 'malignant' column not found in predict_proba output.")

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_proba.astype(float).values,
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare results DataFrame
    results = test[['image_name']].copy()
    results['malignancy_probability'] = malignancy_proba.values

    # Save results in the same format/extension as test.csv
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    results.to_csv(results_path, index=False)

    # 7. Validation checks
    assert len(results) == len(test), "Number of predictions does not match test set."
    assert all(results['image_name'].values == test['image_name'].values), \
        "Prediction file does not preserve test data indices/order."
    expected_cols = ['image_name', 'malignancy_probability']
    assert list(results.columns) == expected_cols, \
        f"Output columns {list(results.columns)} do not match expected {expected_cols}."
    assert results_path.endswith('.csv'), "Results file is not CSV."
    assert np.all((results['malignancy_probability'] >= 0) & (results['malignancy_probability'] <= 1)), \
        "Predicted probabilities are not in [0, 1]."

    # 8. Validation metric on held-out set
    try:
        val_pred_proba = predictor.predict_proba(val_data)
        val_true = val_data['label'].map(lambda x: 1 if x == 'malignant' else 0).values
        if 'malignant' in val_pred_proba.columns:
            val_pred = val_pred_proba['malignant'].values
        else:
            val_pred = val_pred_proba.iloc[:, -1].values
        val_auc = roc_auc_score(val_true, val_pred)
        print(f"Validation AUROC: {val_auc:.5f}")

        # Fairness check: AUROC by skin_tone group (if available)
        if 'skin_tone' in val_data.columns:
            skin_tones = val_data['skin_tone'].unique()
            group_aucs = {}
            for tone in skin_tones:
                idx = val_data['skin_tone'] == tone
                if idx.sum() < 2:
                    continue
                group_auc = roc_auc_score(val_true[idx], val_pred[idx])
                group_aucs[tone] = group_auc
            if group_aucs:
                print("Validation AUROC by skin_tone:", group_aucs)
                if len(group_aucs) > 1:
                    gap = max(group_aucs.values()) - min(group_aucs.values())
                    print(f"AUROC gap between skin_tones: {gap:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Results saved to: {results_path}")
    print(f"Model saved to: {model_folder}")