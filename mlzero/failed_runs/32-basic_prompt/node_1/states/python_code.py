"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a simple image feature-based machine learning model to predict the probability
that a skin lesion is malignant, using a small dataset of skin lesion images. It preprocesses the data,
extracts basic image features, trains a classifier, saves the model, predicts malignancy probabilities
for the test set, and validates output integrity.

Installation requirements (run before executing this script):
    pip install pandas scikit-learn pillow joblib

Assumptions:
- Training data CSV: /home/anri21/be-fair/mlzero/basic_prompt_data/mydataset.csv
- Images are referenced by 'image_name' column and are located in /home/anri21/be-fair/mlzero/basic_prompt_data
- Test data CSV: /home/anri21/be-fair/mlzero/32-basic_prompt/node_1/input/test.csv
- Output directory: /home/anri21/be-fair/mlzero/32-basic_prompt/node_1/output

The script:
- Drops NA labels from training data only.
- Removes index column if present.
- Extracts basic features from images (mean, std, min, max pixel values).
- Trains a logistic regression classifier.
- Saves the trained model in a timestamped folder under the output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Validates output integrity and prints AUROC on a held-out validation set.

Author: AutoML Agent
"""

# Installation instructions (uncomment if running in a fresh environment)
# !pip install pandas scikit-learn pillow joblib

import os
import random
import time
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

def extract_image_features(image_path):
    """
    Extracts simple statistical features from an image.
    Returns a dict of features.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img) / 255.0  # Normalize pixel values
        features = {
            'mean_r': arr[..., 0].mean(),
            'mean_g': arr[..., 1].mean(),
            'mean_b': arr[..., 2].mean(),
            'std_r': arr[..., 0].std(),
            'std_g': arr[..., 1].std(),
            'std_b': arr[..., 2].std(),
            'min_r': arr[..., 0].min(),
            'min_g': arr[..., 1].min(),
            'min_b': arr[..., 2].min(),
            'max_r': arr[..., 0].max(),
            'max_g': arr[..., 1].max(),
            'max_b': arr[..., 2].max(),
        }
    except Exception as e:
        # If image cannot be loaded, fill with NaNs
        features = {k: np.nan for k in [
            'mean_r','mean_g','mean_b','std_r','std_g','std_b',
            'min_r','min_g','min_b','max_r','max_g','max_b'
        ]}
    return features

if __name__ == "__main__":
    # Paths
    DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
    OUTPUT_DIR = "/home/anri21/be-fair/mlzero/32-basic_prompt/node_1/output"
    TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")
    TEST_CSV = "/home/anri21/be-fair/mlzero/32-basic_prompt/node_1/input/test.csv"
    IMAGE_FOLDER = DATA_DIR  # Images are in the same folder as the CSV

    # 1. Data Loading and Preprocessing
    train_df = pd.read_csv(TRAIN_CSV)
    # Remove index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    # Remove training samples without valid labels (drop NA from 'label' column)
    train_df = train_df.dropna(subset=['label'])
    # Map 'image_name' to absolute path
    train_df['image_path'] = train_df['image_name'].apply(lambda x: os.path.join(IMAGE_FOLDER, x) if not os.path.isabs(x) else x)
    # Map 'malignant' -> 1, everything else -> 0
    train_df['label'] = (train_df['label'].str.lower() == 'malignant').astype(int)

    # 2. Validation Split (10% holdout)
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1 if len(train_df) > 1 else 0,  # Avoid error if only 1 row
        random_state=42,
        stratify=train_df['label'] if len(train_df['label'].unique()) > 1 else None
    )

    # 3. Feature Extraction for Training and Validation
    def featurize_df(df):
        feats = []
        for path in df['image_path']:
            feats.append(extract_image_features(path))
        feats_df = pd.DataFrame(feats, index=df.index)
        return feats_df

    X_train = featurize_df(train_data)
    y_train = train_data['label'].values
    X_val = featurize_df(val_data) if len(val_data) > 0 else None
    y_val = val_data['label'].values if len(val_data) > 0 else None

    # Fill missing values (if any image failed to load)
    X_train = X_train.fillna(X_train.mean())
    if X_val is not None:
        X_val = X_val.fillna(X_train.mean())

    # 4. Model Training
    # Prepare output model folder with random timestamp
    timestamp = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"ml_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Use logistic regression for probability output
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    # Save model
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

    # 5. Load Test Data
    test_df = pd.read_csv(TEST_CSV)
    test_index = test_df.index.copy()
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(IMAGE_FOLDER, x) if not os.path.isabs(x) else x)

    # 6. Feature Extraction for Test
    X_test = featurize_df(test_df)
    X_test = X_test.fillna(X_train.mean())

    # 7. Prediction
    malignancy_proba = clf.predict_proba(X_test)[:, 1]
    # Prepare output DataFrame
    results_df = test_df.copy()
    results_df['label'] = malignancy_proba
    # Only keep columns required in the output (match test format)
    output_columns = []
    if 'image_name' in test_df.columns:
        output_columns.append('image_name')
    output_columns.append('label')
    results_df = results_df[output_columns]
    results_df.index = test_index

    # 8. Save Results
    test_ext = os.path.splitext(TEST_CSV)[-1].lower()
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    if test_ext == '.csv':
        results_df.to_csv(result_path, index=False)
    elif test_ext in ['.tsv', '.txt']:
        results_df.to_csv(result_path, sep='\t', index=False)
    else:
        results_df.to_csv(result_path, index=False)

    # 9. Validation Checks
    # 9.1 Check number of rows matches test set
    assert len(results_df) == len(test_df), "Number of predictions does not match number of test samples."
    # 9.2 Check indices match
    assert all(results_df.index == test_index), "Prediction indices do not match test data indices."
    # 9.3 Check column names match requirements
    assert results_df.columns.tolist() == output_columns, f"Output columns {results_df.columns.tolist()} do not match required columns {output_columns}."
    # 9.4 Check output format
    if test_ext == '.csv':
        check_df = pd.read_csv(result_path)
    elif test_ext in ['.tsv', '.txt']:
        check_df = pd.read_csv(result_path, sep='\t')
    else:
        check_df = pd.read_csv(result_path)
    assert check_df.shape[0] == test_df.shape[0], "Saved prediction file row count mismatch."
    assert check_df.columns.tolist() == output_columns, "Saved prediction file columns mismatch."
    # 9.5 Sanity check: probabilities between 0 and 1
    assert np.all((results_df['label'] >= 0) & (results_df['label'] <= 1)), "Predicted probabilities are not in [0, 1]."

    # 10. Validation Metric on Holdout Set
    try:
        if X_val is not None and len(X_val) > 0 and len(np.unique(y_val)) > 1:
            val_pred_proba = clf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred_proba)
            print(f"Validation AUROC: {val_auc:.4f}")
        else:
            print("Validation set too small or only one class present; skipping AUROC computation.")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {result_path}")