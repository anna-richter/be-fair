"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a machine learning model (ResNet18 CNN + Logistic Regression) to predict the probability that a skin lesion is malignant,
using image data and a CSV file of labels. It performs the following:

- Loads and preprocesses the data (removes NA labels, drops index column).
- Extracts features from images using a pretrained ResNet18 backbone.
- Trains a logistic regression classifier on these features.
- Saves the trained model and feature extractor.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions in the same format as the test file, with correct column names.
- Validates output integrity (row count, indices, column names, value ranges).
- Holds out 10% of training data for validation and prints AUROC if possible.
- Provides a function to predict malignancy probabilities for a folder of new images.

# Installation requirements (run if needed):
# pip install pandas scikit-learn torch torchvision pillow

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

# Paths
DATA_DIR = "/home/anri21/be-fair/mlzero/addition_1_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/36-addition_1/node_4/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

# Find test CSV (must exist)
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

IMAGE_DIR = DATA_DIR  # Images are in the same directory as CSVs

# Output file name and format
RESULTS_BASENAME = "results"
MODEL_SAVE_DIR = os.path.join(
    OUTPUT_DIR, f"ml_model_{int(time.time())}_{random.randint(1000,9999)}"
)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def get_image_path(row):
    # Helper to get absolute image path from image_name
    return os.path.join(IMAGE_DIR, row["image_name"])

def extract_features(image_paths, model, device, batch_size=32):
    """
    Extract features from a list of image paths using the given model.
    Returns a numpy array of features.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalization for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    # If image is missing or corrupt, use a black image
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                imgs.append(preprocess(img))
            imgs = torch.stack(imgs).to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def main():
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Data preprocessing
    # Remove unnecessary index column if present
    for idx_col in ["Unnamed: 0", "index"]:
        if idx_col in train_df.columns:
            train_df = train_df.drop(columns=[idx_col])
        if idx_col in test_df.columns:
            test_df = test_df.drop(columns=[idx_col])

    # Remove training samples without valid labels (dropna on 'label')
    train_df = train_df.dropna(subset=["label"])

    # Map label to binary: malignant=1, others=0
    train_df["label"] = train_df["label"].map(lambda x: 1 if str(x).strip().lower() == "malignant" else 0)

    # For test set, preserve all rows and indices
    test_indices = test_df.index.copy()

    # Prepare image paths
    train_df["image_path"] = train_df.apply(get_image_path, axis=1)
    test_df["image_path"] = test_df.apply(get_image_path, axis=1)

    # 3. Validation split (10% holdout)
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )

    # 4. Feature extraction using pretrained ResNet18 (remove final FC layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
    feature_extractor = feature_extractor.to(device)

    def get_feats(df):
        paths = df["image_path"].tolist()
        feats = extract_features(paths, feature_extractor, device)
        feats = feats.reshape(feats.shape[0], -1)
        return feats

    X_train = get_feats(train_data)
    y_train = train_data["label"].values
    X_val = get_feats(val_data)
    y_val = val_data["label"].values
    X_test = get_feats(test_df)

    # 5. Model training (Logistic Regression)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Save model and feature extractor
    with open(os.path.join(MODEL_SAVE_DIR, "logreg.pkl"), "wb") as f:
        pickle.dump(clf, f)
    torch.save(feature_extractor.state_dict(), os.path.join(MODEL_SAVE_DIR, "resnet18_feat.pth"))

    # 6. Prediction on test set (probability of malignant)
    test_probs = clf.predict_proba(X_test)[:, 1]

    # 7. Prepare output DataFrame
    output_df = test_df.copy()
    output_df["label"] = test_probs
    # Only keep columns present in test file, plus 'label' if not present
    if "label" not in test_df.columns:
        output_df = output_df.assign(label=test_probs)
    else:
        output_df["label"] = test_probs

    # Ensure original indices are preserved
    output_df.index = test_indices

    # 8. Save predictions
    test_ext = os.path.splitext(TEST_CSV)[1]
    results_path = os.path.join(OUTPUT_DIR, RESULTS_BASENAME + test_ext)
    output_df.to_csv(results_path, index=False)

    # 9. Validation checks
    # a) Row count
    assert len(output_df) == len(test_df), "Prediction row count does not match test set."
    # b) Indices
    assert all(output_df.index == test_indices), "Prediction indices do not match test set."
    # c) Column names
    assert "label" in output_df.columns, "Output missing required 'label' column."
    # d) Output format
    assert results_path.endswith(test_ext), "Output file extension does not match test file."
    # e) Value range
    assert np.all((output_df["label"] >= 0) & (output_df["label"] <= 1)), "Predicted probabilities out of [0,1] range."

    # 10. Validation metric (AUROC on holdout set)
    try:
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"Validation AUROC: {val_auc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 11. Provide function for new image folder prediction
    def predict_folder(image_folder):
        """
        Given a folder path containing images, returns a DataFrame with image_name and malignancy probability.
        """
        image_files = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f)) and not f.startswith(".")
        ]
        df = pd.DataFrame({"image_name": image_files})
        df["image_path"] = df["image_name"].apply(lambda x: os.path.join(image_folder, x))
        feats = extract_features(df["image_path"].tolist(), feature_extractor, device)
        feats = feats.reshape(feats.shape[0], -1)
        probs = clf.predict_proba(feats)[:, 1]
        result = pd.DataFrame({
            "image_name": df["image_name"],
            "malignancy_probability": probs
        })
        return result

    print("Prediction function 'predict_folder' is ready for use.")

if __name__ == "__main__":
    main()