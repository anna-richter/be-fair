"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a machine learning model to predict the probability of malignancy for skin lesion images,
using both image features and tabular metadata (skin_tone, alternative_skin_tone). It preprocesses the data,
extracts features, trains a model, saves it, predicts on the test set, and writes results in the required format.

- Input: train.csv, test.csv, and JPEG images in MyImages/
- Output: Probability predictions for test images, saved as "results" in the same format as test.csv
- Model and outputs are saved to: /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/output

Installation requirements (run in bash script before running this Python script):
    pip install --upgrade pip
    pip install pandas scikit-learn pillow tqdm joblib

Usage:
    Place this script in the working directory and run with Python 3.7+.
    Ensure all data files are present at the specified paths.

Design notes:
- Drops NA labels from train only, never from test.
- Removes index column 'Unnamed: 0' if present.
- Uses both image and tabular features for fairness.
- Holds out 10% of training data for validation if no validation set is provided.
- Ensures output file matches test.csv format and indices exactly.
- Performs validation checks at the end.
- Efficiently utilizes available CPUs for feature extraction and model training.
"""

import os
import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm

from PIL import Image
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import joblib

# Constants
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
IMAGE_DIR = os.path.join(DATA_DIR, "MyImages")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/48-addition_2"
# end change
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results")
MODEL_DIR = os.path.join(OUTPUT_DIR, f"model_{uuid.uuid4().hex}")

N_JOBS = min(32, os.cpu_count())  # Use up to 32 CPUs for parallel feature extraction

# start change
# def get_image_path(image_name):
#     # Returns absolute path to image file given image_name (without extension)
#     return os.path.join(IMAGE_DIR, f"{image_name}.jpg")
def get_image_path(image_name, ext=".jpg"):
    return os.path.join(IMAGE_DIR, f"{image_name}{ext}")
# end change

def extract_image_features(image_path, resize=(128, 128)):
    """
    Extract simple image features: mean and std of RGB channels, plus resized flattened pixels.
    This is a compromise between speed and informativeness.
    """
    try:
        img = Image.open(image_path).convert('RGB').resize(resize)
        arr = np.array(img) / 255.0
        # Channel-wise mean and std
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        # Flattened, downsampled image (for more info)
        flat = arr.flatten()
        features = np.concatenate([means, stds, flat])
        return features
    except Exception:
        # If image is missing or unreadable, return NaNs
        return np.full(3 + 3 + resize[0]*resize[1]*3, np.nan)

# start change
# def extract_features_df(df, resize=(128, 128)):
def extract_features_df(df, resize=(128, 128), ext=".jpg"):
# end change
    """
    Extract features for all images in a DataFrame.
    Returns a DataFrame with image features and tabular features.
    Handles missing tabular columns in test set gracefully.
    """
    # start change
    # image_paths = df['image_name'].apply(get_image_path).tolist()
    image_paths = df['image_name'].apply(lambda x: get_image_path(x, ext=ext)).tolist()
    # end change
    # Parallel feature extraction
    features = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(extract_image_features)(p, resize=resize) for p in tqdm(image_paths, desc="Extracting image features")
    )
    features = np.stack(features)
    # Tabular features: skin_tone, alternative_skin_tone (handle missing columns)
    tabular_cols = []
    for col in ['skin_tone', 'alternative_skin_tone']:
        if col in df.columns:
            tabular_cols.append(col)
    if tabular_cols:
        tabular = df[tabular_cols].fillna(-1).to_numpy()
    else:
        # If missing, fill with -1s
        tabular = np.full((len(df), 2), -1)
    # If only one tabular column, pad to 2 columns for consistency
    if tabular.shape[1] == 1:
        tabular = np.concatenate([tabular, np.full((len(df), 1), -1)], axis=1)
    all_features = np.concatenate([features, tabular], axis=1)
    return all_features

def main():
    # 1. Data Loading and Preprocessing
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove index column if present
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Drop training samples with missing labels (but not from test)
    train = train.dropna(subset=['label'])

    # Map label to binary: malignant=1, benign=0
    # From the sample, label is 'non-neoplastic' for benign, so we need to map accordingly.
    label_map = {}
    unique_labels = train['label'].unique()
    for val in unique_labels:
        if isinstance(val, str) and 'malig' in val.lower():
            label_map[val] = 1
        else:
            label_map[val] = 0
    train['label'] = train['label'].map(label_map)

    # 2. Validation Split (if no validation set is provided)
    train_data, val_data = train_test_split(
        train, test_size=0.1, random_state=42, stratify=train['label']
    )

    # 3. Feature Extraction
    print("Extracting features for training data...")
    X_train = extract_features_df(train_data)
    y_train = train_data['label'].values

    print("Extracting features for validation data...")
    X_val = extract_features_df(val_data)
    y_val = val_data['label'].values

    print("Extracting features for test data...")
    # start change
    # X_test = extract_features_df(test)  # original (.jpg)
    X_test = extract_features_df(test, ext=".png")
    # end change

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5. Model Training
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Use a RandomForestClassifier for speed and robustness
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        n_jobs=N_JOBS,
        random_state=42,
        class_weight='balanced_subsample'
    )
    clf.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump({'model': clf, 'scaler': scaler}, os.path.join(MODEL_DIR, "rf_model_and_scaler.joblib"))

    # 6. Prediction
    # Predict probabilities (malignancy probability)
    proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (malignant)

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": proba.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare output DataFrame: must match test.csv format and indices
    results_df = test[['image_name']].copy()
    results_df['malignancy_probability'] = proba

    # Save results in the same format and extension as test.csv (i.e., CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)

    # 7. Validation Checks
    # a) Check that output file has same number of rows as test set
    pred_df = pd.read_csv(RESULTS_FILE)
    assert len(pred_df) == len(test), f"Prediction file has {len(pred_df)} rows, expected {len(test)}"
    # b) Check that indices match (image_name order)
    assert all(pred_df['image_name'] == test['image_name']), "Image names in prediction file do not match test set order"
    # c) Check that columns are correct
    assert list(pred_df.columns) == ['image_name', 'malignancy_probability'], f"Prediction columns {list(pred_df.columns)} do not match required ['image_name', 'malignancy_probability']"
    # d) Check that probabilities are floats in [0,1]
    assert np.all((pred_df['malignancy_probability'] >= 0) & (pred_df['malignancy_probability'] <= 1)), "Probabilities are not all in [0,1]"

    # 8. Validation Metric (on held-out validation set)
    try:
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_proba)
        print(f"Validation ROC-AUC: {val_score:.4f}")
        # Also print AUROC gap between light and dark skin tones
        if 'skin_tone' in val_data.columns:
            light_mask = val_data['skin_tone'] == 1
            dark_mask = val_data['skin_tone'] == 3
            if light_mask.sum() > 0 and dark_mask.sum() > 0:
                light_auc = roc_auc_score(y_val[light_mask], val_proba[light_mask])
                dark_auc = roc_auc_score(y_val[dark_mask], val_proba[dark_mask])
                print(f"Light skin ROC-AUC: {light_auc:.4f}")
                print(f"Dark skin ROC-AUC: {dark_auc:.4f}")
                print(f"AUROC gap (light - dark): {light_auc - dark_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()