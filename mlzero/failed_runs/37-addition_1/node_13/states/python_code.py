"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a machine learning model (using image features) to predict the probability that a skin lesion is malignant.
It:
- Loads and preprocesses the training data (removes NA labels, drops index column).
- Extracts features from lesion images using a pretrained CNN (ResNet18).
- Trains a binary classifier (Logistic Regression) on these features.
- Saves the trained model to a timestamped folder in the specified output directory.
- Makes predictions (probabilities) on the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation using a 10% holdout from the training data and prints AUROC.
- Includes validation checks to ensure output correctness.

# Installation (if needed):
# !pip install pandas scikit-learn torch torchvision pillow

Data paths and output directory are hardcoded as per instructions.
"""

import os
import random
import time
import pickle
import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ==== CONFIGURATION ====
DATA_DIR = "/home/anri21/be-fair/mlzero/addition_1_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/37-addition_1/node_13/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

def get_test_csv_path():
    # Try to find the test file in the data directory
    for fname in os.listdir(DATA_DIR):
        if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
            return os.path.join(DATA_DIR, fname)
    raise FileNotFoundError("Test CSV file not found in data directory.")

def make_output_model_dir():
    # Create a random timestamped model directory in OUTPUT_DIR
    ts = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"ml_model_{ts}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def extract_image_features(image_paths, device, batch_size=32):
    """
    Extracts features from images using pretrained ResNet18 (removing final FC layer).
    Returns a numpy array of shape (num_images, feature_dim).
    """
    # Preprocessing pipeline for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Load pretrained ResNet18, remove final FC
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]  # Remove last FC
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    feature_extractor.to(device)

    features = []
    n = len(image_paths)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    # If image is missing or unreadable, use a black image
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                imgs.append(preprocess(img))
            imgs = torch.stack(imgs).to(device)
            feats = feature_extractor(imgs).squeeze(-1).squeeze(-1)  # (B, 512, 1, 1) -> (B, 512)
            features.append(feats.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

if __name__ == "__main__":
    # ==== 1. Load and preprocess data ====
    train_df = pd.read_csv(TRAIN_CSV)
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    # Remove training samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])
    train_df = train_df.reset_index(drop=True)

    # Map 'label' to binary: malignant=1, benign=0 (assuming 'malignant' and 'benign' are possible values)
    # If 'non-neoplastic' is present, treat as benign (0)
    label_map = {'malignant': 1, 'benign': 0, 'non-neoplastic': 0}
    train_df['label'] = train_df['label'].map(label_map)
    if train_df['label'].isnull().any():
        raise ValueError("Unknown label values found in training data.")

    # Add absolute image path column
    train_df['image_path'] = train_df['image_name'].apply(lambda x: os.path.join(DATA_DIR, x))

    # ==== 2. Prepare test data ====
    test_csv_path = get_test_csv_path()
    test_df = pd.read_csv(test_csv_path)
    test_index = test_df.index.copy()  # Save original indices for validation

    # Remove unnecessary index column if present
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])

    # Add absolute image path column
    test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(DATA_DIR, x))

    # ==== 3. Split train/validation ====
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df['label']
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # ==== 4. Extract image features ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Extracting train features...")
    X_train = extract_image_features(train_data['image_path'].tolist(), device)
    y_train = train_data['label'].values

    print("Extracting validation features...")
    X_val = extract_image_features(val_data['image_path'].tolist(), device)
    y_val = val_data['label'].values

    print("Extracting test features...")
    X_test = extract_image_features(test_df['image_path'].tolist(), device)

    # ==== 5. Train model ====
    model_dir = make_output_model_dir()
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Save model
    with open(os.path.join(model_dir, "logreg_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # ==== 6. Predict on test set ====
    y_test_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (malignant)

    # ==== 7. Prepare output ====
    # Output format: same as test file, but with a column for malignancy probability
    output_df = test_df.copy()
    output_df['label'] = y_test_proba

    # Save output in same format as test file (CSV)
    test_ext = os.path.splitext(test_csv_path)[1]
    results_path = os.path.join(OUTPUT_DIR, "results" + test_ext)
    output_df.to_csv(results_path, index=False)

    # ==== 8. Validation checks ====
    # 1. Check number of rows matches test set
    assert len(output_df) == len(test_df), "Number of predictions does not match number of test samples."
    # 2. Check indices preserved
    assert all(output_df.index == test_index), "Test indices not preserved in output."
    # 3. Check output columns: must include 'label'
    assert 'label' in output_df.columns, "Output file missing 'label' column."
    # 4. Check output format: extension matches test file
    assert os.path.splitext(results_path)[1] == test_ext, "Output file extension does not match test file."
    # 5. Check predictions are floats in [0,1]
    assert np.all((output_df['label'] >= 0) & (output_df['label'] <= 1)), "Predicted probabilities not in [0,1]."

    # ==== 9. Validation on holdout set ====
    try:
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_proba)
        print(f"Validation AUROC: {val_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")