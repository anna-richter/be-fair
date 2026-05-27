"""
Skin Lesion Malignancy Probability Prediction Script

This script:
- Loads and preprocesses skin lesion image data for binary classification (malignant vs. non-malignant).
- Extracts features from images using a pretrained CNN (EfficientNet-B3).
- Trains a fairness-aware LightGBM classifier with advanced regularization and bagging to predict malignancy probability.
- Saves the trained model and outputs malignancy probabilities for the test set, preserving original indices and format.
- Performs validation (AUROC) on a held-out validation set if training labels are available.
- Includes robust validation checks on the output file.

Installation requirements (run in bash before running this script):
# pip install pandas numpy scikit-learn lightgbm pillow tqdm torch torchvision timm

Author: AutoML Agent
"""

import os
import uuid
import time
import warnings

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import timm
import torchvision.transforms as transforms

import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# Constants
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/49-addition_3/node_7/output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_path(image_name):
    """Helper to get absolute image path from image_name."""
    return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

def map_label_to_binary(label):
    """Maps string label to binary: malignant=1, non-malignant=0."""
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label == "benign":
        return 0
    elif label == "non-neoplastic":
        return 0
    elif label == "neoplastic":
        return 1
    else:
        return np.nan

def compute_class_weights(df, label_col, group_col=None):
    """
    Compute class weights for fairness.
    If group_col is provided, computes group-balanced weights.
    """
    if group_col is None:
        counts = df[label_col].value_counts()
        weights = {int(k): float(1.0/v) for k, v in counts.items()}
        norm = sum(weights.values())
        weights = {int(k): float(v/norm) for k, v in weights.items()}
        return weights
    else:
        groups = df[group_col].unique()
        class_weights = []
        for g in groups:
            sub = df[df[group_col]==g]
            counts = sub[label_col].value_counts()
            weights = {int(k): float(1.0/v) for k, v in counts.items()}
            norm = sum(weights.values())
            weights = {int(k): float(v/norm) for k, v in weights.items()}
            class_weights.append(weights.get(1, 0.0))
        avg_weight_1 = float(np.mean(class_weights))
        avg_weight_0 = float(1 - avg_weight_1)
        return {0: avg_weight_0, 1: avg_weight_1}

def extract_image_features(image_paths, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extracts features from images using pretrained EfficientNet-B3 (global avgpool output).
    Returns a numpy array of shape (len(image_paths), 1536).
    """
    # Use EfficientNet-B3 from timm for better feature representation
    model = timm.create_model('efficientnet_b3', pretrained=True)
    model.eval()
    model.to(device)
    # Remove classifier head, keep global pooling
    if hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()

    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features (EffNetB3)"):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    img = Image.fromarray(np.zeros((300,300,3), dtype=np.uint8))
                imgs.append(preprocess(img))
            imgs = torch.stack(imgs).to(device)
            feats = model(imgs)
            if feats.ndim > 2:
                feats = feats.mean(dim=[2,3])  # global avgpool if needed
            features.append(feats.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def validate_output(pred_path, test_df, pred_col, file_ext):
    """
    Validation checks:
    - Output file exists and loads
    - Output has same number of rows as test_df
    - Output indices match test_df indices
    - Output columns match requirements
    - Output format matches test file
    - Output predictions are floats in [0,1]
    """
    assert os.path.exists(pred_path), f"Prediction file {pred_path} does not exist."
    if file_ext == ".csv":
        pred_df = pd.read_csv(pred_path)
    elif file_ext == ".parquet":
        pred_df = pd.read_parquet(pred_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    # Check row count
    assert len(pred_df) == len(test_df), f"Prediction rows {len(pred_df)} != test rows {len(test_df)}"
    # Check index alignment
    assert all(pred_df.index == test_df.index), "Prediction indices do not match test indices"
    # Check columns
    assert pred_col in pred_df.columns, f"Prediction column {pred_col} missing in output"
    # Check format
    assert pred_df.shape[1] == test_df.shape[1], f"Prediction columns {pred_df.columns} != test columns {test_df.columns}"
    # Check prediction values
    preds = pred_df[pred_col].values
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions not in [0,1] range"
    print("Validation checks passed.")

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Remove unnecessary index column if present
    for df in [train_df, test_df]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # 3. Preprocess train: drop NA labels, map to binary
    train_df["label"] = train_df["label"].apply(map_label_to_binary)
    train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)
    train_df["label"] = train_df["label"].astype(int)

    # 4. Add image path columns
    train_df["image"] = train_df["image_name"].apply(get_image_path)
    test_df["image"] = test_df["image_name"].apply(get_image_path)

    # 5. Prepare test set: preserve original indices
    test_df_orig = test_df.copy()
    test_df = test_df.reset_index(drop=True)

    # 6. Fairness-aware class weights (group reweighting by skin_tone)
    if "skin_tone" in train_df.columns and train_df["skin_tone"].notna().any():
        class_weights = compute_class_weights(train_df, "label", group_col="skin_tone")
    else:
        class_weights = compute_class_weights(train_df, "label")
    weights_list = [float(class_weights.get(0, 0.5)), float(class_weights.get(1, 0.5))]

    # 7. Hold out validation set (10%)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.10, random_state=42, stratify=train_df["label"]
    )
    train_df2 = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # 8. Extract image features (EfficientNet-B3, 1536-dim)
    train_features = extract_image_features(train_df2["image"].tolist())
    val_features = extract_image_features(val_df["image"].tolist())
    test_features = extract_image_features(test_df["image"].tolist())

    # 9. Train LightGBM classifier (advanced regularization and bagging)
    lgb_train = lgb.Dataset(train_features, label=train_df2["label"].values)
    lgb_val = lgb.Dataset(val_features, label=val_df["label"].values, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "is_unbalance": False,
        "scale_pos_weight": weights_list[1] / weights_list[0] if weights_list[0] > 0 else 1.0,
        "learning_rate": 0.025,
        "num_leaves": 64,
        "max_depth": 8,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 2,
        "lambda_l1": 1.0,
        "lambda_l2": 2.0,
        "seed": 42,
        "n_jobs": 4,
    }

    # Save model to unique timestamped folder
    model_folder = os.path.join(
        OUTPUT_DIR, f"ml_model_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, "lgb_model.pkl")

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=True)]

    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=400,
        callbacks=callbacks,
    )

    joblib.dump(gbm, model_path)

    # 10. Validation (AUROC)
    try:
        val_pred = gbm.predict(val_features, num_iteration=gbm.best_iteration)
        val_auc = roc_auc_score(val_df["label"].values, val_pred)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 11. Prediction on test set
    test_pred = gbm.predict(test_features, num_iteration=gbm.best_iteration)
    test_pred = np.clip(test_pred, 0, 1)

    # 12. Prepare output DataFrame
    result_df = test_df_orig.copy()
    result_df["label"] = test_pred
    keep_cols = [col for col in test_df_orig.columns if col not in ["Unnamed: 0", "image"]] + ["label"]
    result_df = result_df[keep_cols]

    # 13. Save results in same format/extension as test.csv
    test_ext = os.path.splitext(TEST_CSV)[1].lower()
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    if test_ext == ".csv":
        result_df.to_csv(result_path, index=False)
    elif test_ext == ".parquet":
        result_df.to_parquet(result_path, index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {test_ext}")

    print(f"Saved predictions to {result_path}")

    # 14. Validation checks
    validate_output(result_path, test_df_orig, "label", test_ext)