"""
Skin Lesion Malignancy Prediction Script using AutoGluon MultiModal

This script performs binary image classification (malignant vs. benign) on skin lesion images.
It:
- Loads and preprocesses the data (removes NA labels, drops index columns)
- Trains an AutoGluon MultiModalPredictor on the training set (with 10% held out for validation)
- Predicts malignancy probability (float in [0,1]) for each test image, preserving original indices and format
- Saves the trained model and prediction results to the specified output directory
- Performs validation checks on the output and prints AUROC on the held-out validation set

Installation requirements (run if needed):
    pip install --upgrade pip
    pip install autogluon.multimodal

Usage:
    Place this script in any location and run it. It expects the data at:
        /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/
    and will write all outputs to:
        /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/output/

Author: AutoML Agent
"""

import os
import uuid
import time
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

from autogluon.multimodal import MultiModalPredictor

# =========================
# Paths and Constants
# =========================
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMG_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/31-basic_prompt"
# end change
RESULTS_FILENAME = "results"
MODEL_DIR = os.path.join(
    OUTPUT_DIR, f"autogluon_model_{int(time.time())}_{uuid.uuid4().hex[:8]}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _drop_index_col(df):
    # Remove index column if present (e.g., 'Unnamed: 0' or unnamed first column)
    if "Unnamed: 0" in df.columns:
        return df.drop(columns=["Unnamed: 0"])
    if df.columns[0].lower() in ["", "unnamed: 0"]:
        return df.iloc[:, 1:]
    return df

def _get_image_path(image_name):
    # Compose absolute path to image file
    return os.path.abspath(os.path.join(IMG_DIR, f"{image_name}.jpg"))

def _map_label_to_binary(label):
    # Map label to binary: malignant=1, benign=0
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    else:
        return 0

def _get_output_extension(test_csv_path):
    base, ext = os.path.splitext(test_csv_path)
    return ext

def _get_result_path(test_csv_path):
    ext = _get_output_extension(test_csv_path)
    return os.path.join(OUTPUT_DIR, RESULTS_FILENAME + ext)

def _get_test_index_col(test_df):
    if "Unnamed: 0" in test_df.columns:
        return "Unnamed: 0"
    if test_df.columns[0].lower() in ["", "unnamed: 0"]:
        return test_df.columns[0]
    return None

def _get_label_col(train_df):
    for col in train_df.columns:
        if col.lower() == "label":
            return col
    raise ValueError("No label column found in training data.")

def _get_image_col(df):
    for col in df.columns:
        if "image" in col.lower():
            return col
    raise ValueError("No image column found in data.")

def _ensure_absolute_image_paths(df, image_col):
    df = df.copy()
    df[image_col] = df[image_col].apply(_get_image_path)
    return df

def _validate_predictions(
    test_df,
    pred_df,
    train_df,
    result_path,
    ext,
    prob_col,
):
    # 1. Check number of rows
    assert len(pred_df) == len(test_df), (
        f"Prediction file has {len(pred_df)} rows, "
        f"but test file has {len(test_df)} rows."
    )
    # 2. Check indices match
    test_id_col = _get_test_index_col(test_df)
    if test_id_col is not None:
        assert (pred_df[test_id_col].values == test_df[test_id_col].values).all(), (
            "Prediction file indices do not match test file indices."
        )
    # 3. Check columns
    expected_cols = list(test_df.columns)
    expected_cols = [c for c in expected_cols if "image" not in c.lower()]
    expected_cols.append(prob_col)
    pred_cols = list(pred_df.columns)
    assert (
        prob_col in pred_cols
    ), f"Prediction file missing required column: {prob_col}"
    # 4. Check output format
    if ext == ".csv":
        pd.read_csv(result_path)
    elif ext == ".tsv":
        pd.read_csv(result_path, sep="\t")
    # 5. Check probability values are in [0,1]
    probs = pred_df[prob_col].values
    assert np.all((probs >= 0) & (probs <= 1)), (
        "Predicted probabilities are not all in [0,1]."
    )
    print("Validation checks passed.")

if __name__ == "__main__":
    # =========================
    # 1. Load Data
    # =========================
    train_df = pd.read_csv(TRAIN_CSV)
    # start change
    # test_df = pd.read_csv(TEST_CSV)
    test_df = pd.read_csv(TEST_CSV, dtype={"image_name": str})
    # end change

    # Remove unnecessary index columns
    train_df = _drop_index_col(train_df)
    test_df = _drop_index_col(test_df)

    # Remove training samples with missing labels (do NOT drop from test)
    label_col = _get_label_col(train_df)
    train_df = train_df[~train_df[label_col].isna()].reset_index(drop=True)

    # Map label to binary (malignant=1, else 0)
    train_df[label_col] = train_df[label_col].apply(_map_label_to_binary)

    # Get image column
    image_col = _get_image_col(train_df)
    # Replace image_name with absolute path
    train_df = _ensure_absolute_image_paths(train_df, image_col)
    # start change
    # test_df = _ensure_absolute_image_paths(test_df, image_col)  # original (.jpg)
    test_df = test_df.copy()
    test_df[image_col] = test_df[image_col].apply(
        lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
    )
    # end change

    # =========================
    # 2. Train/Validation Split
    # =========================
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df[label_col]
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # =========================
    # 3. Model Training
    # =========================
    predictor = MultiModalPredictor(
        label=label_col, path=MODEL_DIR, problem_type="binary"
    )
    predictor.fit(
        train_data=train_data,
        time_limit=None,
    )

    # =========================
    # 4. Prediction on Test Set
    # =========================
    # Ensure test_df has all columns required by the model except label_col
    missing_cols = [col for col in train_data.columns if col not in test_df.columns and col != label_col]
    for col in missing_cols:
        test_df[col] = np.nan
    # Ensure column order matches train_data (except label_col)
    test_pred_input = test_df[[c for c in train_data.columns if c != label_col]]

    # Predict probabilities for class 1 (malignant)
    proba_df = predictor.predict_proba(test_pred_input)
    if 1 in proba_df.columns:
        malignancy_probs = proba_df[1].values
    else:
        malignancy_probs = proba_df["1"].values

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test_df["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_probs.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # =========================
    # 5. Prepare Output DataFrame
    # =========================
    output_df = test_df.copy()
    output_df = output_df.loc[:, [c for c in output_df.columns if "image" not in c.lower()]]
    output_df[label_col] = malignancy_probs

    # =========================
    # 6. Save Results
    # =========================
    result_path = _get_result_path(TEST_CSV)
    ext = _get_output_extension(TEST_CSV)
    if ext == ".csv":
        output_df.to_csv(result_path, index=False)
    elif ext == ".tsv":
        output_df.to_csv(result_path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported test file extension: {ext}")

    print(f"Predictions saved to: {result_path}")

    # =========================
    # 7. Validation Checks
    # =========================
    _validate_predictions(
        test_df=test_df,
        pred_df=output_df,
        train_df=train_df,
        result_path=result_path,
        ext=ext,
        prob_col=label_col,
    )

    # =========================
    # 8. Validation Metric (AUROC)
    # =========================
    try:
        val_pred_input = val_data[[c for c in train_data.columns if c != label_col]]
        val_pred_proba = predictor.predict_proba(val_pred_input)
        if 1 in val_pred_proba.columns:
            val_probs = val_pred_proba[1].values
        else:
            val_probs = val_pred_proba["1"].values
        val_labels = val_data[label_col].values
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"Validation AUROC: {val_auc:.5f}")
    except Exception as e:
        print(f"Validation failed: {e}")