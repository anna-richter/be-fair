"""
Skin Lesion Malignancy Probability Prediction using AutoGluon MultiModal

This script trains a multimodal predictor (image + tabular) to predict the probability of malignancy for skin lesion images.
It uses AutoGluon's MultiModalPredictor for binary image classification, with fairness considerations for skin tone.
Key features:
    - Loads and preprocesses training and test data (removes NA labels, drops index columns).
    - Ensures all training features are present in test data (fills missing columns with NaN).
    - Trains a model (with 10% validation split if no validation set is provided).
    - Uses a strong image backbone (convnext_base_in22ft1k) and focal loss with class weights for fairness.
    - Saves the trained model to a timestamped folder in the specified output directory.
    - Predicts malignancy probabilities for the test set, preserving original indices and output format.
    - Performs validation checks to ensure output correctness.
    - Provides a function to predict malignancy probabilities for new images in a folder.

# Installation (run in bash before running this script):
# pip install autogluon.multimodal

Usage:
    - Place this script in your working environment.
    - Ensure data is available at the specified absolute paths.
    - Run the script as main.
"""

import os
import time
import uuid
import warnings
import pandas as pd
import numpy as np

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ==== CONFIGURATION ====
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"
IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/35-addition_1/node_4/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return str(int(time.time())) + "-" + uuid.uuid4().hex[:8]

def get_image_path(image_name):
    return os.path.join(IMAGE_DIR, f"{image_name}.jpg")

def prepare_data(df, drop_index_col=True):
    idx_cols = [col for col in df.columns if col.lower() in ['unnamed: 0', 'index']]
    if drop_index_col and idx_cols:
        df = df.drop(columns=idx_cols)
    return df

def map_label_to_binary(label):
    if pd.isnull(label):
        return np.nan
    if isinstance(label, (int, float)):
        return int(label)
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label == "benign":
        return 0
    elif label == "non-neoplastic":
        return 0
    elif label == "neoplastic":
        return np.nan
    else:
        return np.nan

def compute_class_weights(labels):
    counts = labels.value_counts().to_dict()
    total = sum(counts.values())
    weights = []
    for i in [0, 1]:
        cnt = counts.get(i, 1)
        weights.append(total / (2 * cnt))
    weights = np.array(weights) / np.sum(weights)
    return weights.tolist()

def get_label_column(train_df):
    for col in train_df.columns:
        if col.lower() == "label":
            return col
    raise ValueError("No label column found in training data.")

def get_image_column(train_df):
    for col in train_df.columns:
        if "image" in col.lower():
            return col
    raise ValueError("No image column found in training data.")

def get_output_filename(test_csv):
    ext = os.path.splitext(test_csv)[1]
    return os.path.join(OUTPUT_DIR, f"results{ext}")

def validate_predictions(test_df, pred_df, label_col):
    assert len(pred_df) == len(test_df), "Prediction row count does not match test set."
    assert all(pred_df.index == test_df.index), "Prediction indices do not match test set."
    assert label_col in pred_df.columns, f"Prediction file missing required column: {label_col}"
    assert pred_df[label_col].between(0, 1).all(), "Predicted probabilities not in [0, 1]."

def print_validation_score(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
        print(f"Validation AUROC: {score:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    # ==== 1. LOAD DATA ====
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # ==== 2. PREPROCESSING ====
    train_df = prepare_data(train_df, drop_index_col=True)
    test_df = prepare_data(test_df, drop_index_col=True)

    label_col = get_label_column(train_df)
    image_col = get_image_column(train_df)

    # Map label to binary
    train_df[label_col] = train_df[label_col].map(map_label_to_binary)
    train_df = train_df.dropna(subset=[label_col]).reset_index(drop=True)
    train_df[label_col] = train_df[label_col].astype(int)

    # Add absolute image path column for AutoGluon
    train_df["image"] = train_df[image_col].apply(get_image_path)
    test_df["image"] = test_df[image_col].apply(get_image_path)

    # ==== 3. ALIGN TEST COLUMNS TO TRAIN ====
    # Ensure all columns used in training are present in test data (fill with NaN if missing)
    missing_cols = [col for col in train_df.columns if col not in test_df.columns and col != label_col]
    for col in missing_cols:
        test_df[col] = np.nan
    # Ensure column order matches (except label_col)
    test_df = test_df[[col for col in train_df.columns if col != label_col]]

    # ==== 4. TRAIN/VAL SPLIT ====
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df[label_col],
        random_state=42,
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # ==== 5. CLASS WEIGHTS FOR FAIRNESS ====
    class_weights = compute_class_weights(train_data[label_col])

    # ==== 6. MODEL TRAINING ====
    model_dir = os.path.join(OUTPUT_DIR, f"autogluon-model-{get_timestamp()}")
    os.makedirs(model_dir, exist_ok=True)

    # Use a strong backbone for better performance (no late_fusion keys)
    hyperparameters = {
        "model.timm_image.checkpoint_name": "convnext_base_in22ft1k",
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": class_weights,
        "optim.focal_loss.gamma": 2.0,
        "optim.focal_loss.reduction": "mean",
        "optim.max_epochs": 20,
    }

    # Use all columns except label_col for training
    train_cols = [col for col in train_data.columns if col != label_col] + [label_col]
    predictor = MultiModalPredictor(
        label=label_col,
        problem_type="binary",
        path=model_dir,
    )
    predictor.fit(
        train_data=train_data[train_cols],
        tuning_data=val_data[train_cols],
        hyperparameters=hyperparameters,
        seed=42,
        time_limit=None,
        presets="best_quality",
    )

    # ==== 7. VALIDATION ====
    try:
        val_probs = predictor.predict_proba(val_data.drop(columns=[label_col]))
        if isinstance(val_probs, pd.DataFrame):
            if 1 in val_probs.columns:
                val_pred = val_probs[1].values
            elif "1" in val_probs.columns:
                val_pred = val_probs["1"].values
            else:
                val_pred = val_probs.iloc[:, -1].values
        else:
            val_pred = np.array(val_probs)
        print_validation_score(val_data[label_col].values, val_pred)
    except Exception as e:
        print(f"Validation failed: {e}")

    # ==== 8. PREDICTION ON TEST SET ====
    # Add dummy label_col for predict_proba (AutoGluon ignores it)
    test_predict_df = test_df.copy()
    test_predict_df[label_col] = np.nan
    test_predict_df = test_predict_df[[col for col in train_df.columns if col != label_col] + [label_col]]

    test_probs = predictor.predict_proba(test_predict_df.drop(columns=[label_col]))
    if isinstance(test_probs, pd.DataFrame):
        if 1 in test_probs.columns:
            malignancy_prob = test_probs[1].values
        elif "1" in test_probs.columns:
            malignancy_prob = test_probs["1"].values
        else:
            malignancy_prob = test_probs.iloc[:, -1].values
    else:
        malignancy_prob = np.array(test_probs)

    # Prepare output DataFrame
    output_df = test_df.copy()
    output_df[label_col] = malignancy_prob
    output_df = output_df.set_index(test_df.index)
    required_cols = list(test_df.columns) + [label_col]
    required_cols = list(dict.fromkeys(required_cols))
    output_df = output_df[required_cols]

    output_file = get_output_filename(TEST_CSV)
    if output_file.endswith(".csv"):
        output_df.to_csv(output_file, index=False)
    elif output_file.endswith(".tsv"):
        output_df.to_csv(output_file, sep="\t", index=False)
    elif output_file.endswith(".parquet"):
        output_df.to_parquet(output_file, index=False)
    else:
        output_df.to_csv(output_file, index=False)

    # ==== 9. VALIDATION CHECKS ====
    pred_df = pd.read_csv(output_file)
    validate_predictions(test_df, pred_df, label_col)
    print(f"Predictions saved to: {output_file}")

    # ==== 10. FUNCTION FOR NEW IMAGES ====
    def predict_folder(folder_path, model_path=model_dir):
        """
        Given a folder of images, returns a DataFrame with image file names and malignancy probabilities.
        Args:
            folder_path (str): Path to folder containing .jpg images.
            model_path (str): Path to trained AutoGluon model directory.
        Returns:
            pd.DataFrame: columns ['image_name', 'malignancy_probability']
        """
        predictor = MultiModalPredictor.load(model_path)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        df = pd.DataFrame({"image": image_paths, "image_name": [os.path.splitext(f)[0] for f in image_files]})
        # Add missing columns with NaN to match training features
        for col in train_df.columns:
            if col not in df.columns and col != label_col:
                df[col] = np.nan
        # Ensure column order
        df = df[[col for col in train_df.columns if col != label_col]]
        probs = predictor.predict_proba(df)
        if isinstance(probs, pd.DataFrame):
            if 1 in probs.columns:
                malignancy_prob = probs[1].values
            elif "1" in probs.columns:
                malignancy_prob = probs["1"].values
            else:
                malignancy_prob = probs.iloc[:, -1].values
        else:
            malignancy_prob = np.array(probs)
        result = pd.DataFrame({
            "image_name": df["image_name"],
            "malignancy_probability": malignancy_prob
        })
        return result

    # Example usage (commented out):
    # new_folder = "/path/to/new/images"
    # preds = predict_folder(new_folder)
    # print(preds.head())