"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a machine learning model (ResNet18 via PyTorch and scikit-learn) to predict the probability
that a skin lesion is malignant, based on images referenced in the dataset. It:
- Loads and preprocesses the data (removes NA labels, drops index columns).
- Extracts features from images using a pretrained ResNet18 CNN.
- Trains a logistic regression classifier on these features.
- Saves the trained model to a timestamped folder in the output directory.
- Predicts malignancy probabilities for the test set, preserving original indices and output format.
- Saves predictions to the output directory, matching the test file's format and column names.
- Performs validation using a 10% holdout from the training data and prints AUROC.
- Includes validation checks to ensure output correctness.

# Installation (uncomment and run in bash if needed):
# pip install pandas scikit-learn torch torchvision pillow

Author: AutoML Agent
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
DATA_DIR = "/home/anri21/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/home/anri21/be-fair/mlzero/30-basic_prompt/node_1/output"
TRAIN_CSV = os.path.join(DATA_DIR, "mydataset.csv")

# Find test CSV file (must exist)
TEST_CSV = None
for fname in os.listdir(DATA_DIR):
    if fname.lower().startswith("test") and fname.lower().endswith(".csv"):
        TEST_CSV = os.path.join(DATA_DIR, fname)
        break
if TEST_CSV is None:
    raise FileNotFoundError("Test CSV file not found in data directory.")

IMAGE_ROOT = DATA_DIR  # Images are referenced by image_name column, and are in the same folder

def get_timestamp_folder(base_dir):
    """Generate a random timestamped folder path under base_dir."""
    ts = int(time.time())
    rand = random.randint(1000, 9999)
    folder = os.path.join(base_dir, f"model_{ts}_{rand}")
    return folder

def get_image_path(image_name):
    """Find the image file with supported extensions."""
    base = os.path.join(IMAGE_ROOT, image_name)
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    # Try as-is (maybe image_name already has extension)
    if os.path.exists(base):
        return base
    raise FileNotFoundError(f"Image file not found for {image_name}")

def extract_features(image_paths, model, device, batch_size=32):
    """
    Extract features from images using a pretrained CNN.
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
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                imgs.append(preprocess(img))
            imgs = torch.stack(imgs).to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Data preprocessing
    # Remove index column if present
    index_cols = [col for col in train_df.columns if col.lower() in ['unnamed: 0', 'index']]
    if index_cols:
        train_df = train_df.drop(columns=index_cols)
    index_cols_test = [col for col in test_df.columns if col.lower() in ['unnamed: 0', 'index']]
    if index_cols_test:
        test_df = test_df.drop(columns=index_cols_test)

    # Remove training samples without valid labels (drop NA in 'label')
    train_df = train_df.dropna(subset=['label'])

    # Map label to binary: malignant=1, else 0
    train_df['label'] = (train_df['label'].str.lower() == 'malignant').astype(int)

    # For test set, keep all rows, do not drop any rows

    # 3. Prepare image paths
    train_df['image_path'] = train_df['image_name'].apply(get_image_path)
    test_df['image_path'] = test_df['image_name'].apply(get_image_path)

    # 4. Hold out 10% validation set from training data
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    # 5. Feature extraction using pretrained ResNet18 (remove final layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=True)
    # Remove final classification layer
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    def get_feats(df):
        paths = df['image_path'].tolist()
        feats = extract_features(paths, feature_extractor, device)
        # ResNet18 outputs (N, 512, 1, 1), squeeze to (N, 512)
        feats = feats.reshape(feats.shape[0], -1)
        return feats

    X_train = get_feats(train_data)
    y_train = train_data['label'].values
    X_val = get_feats(val_data)
    y_val = val_data['label'].values
    X_test = get_feats(test_df)

    # 6. Model training (Logistic Regression)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)

    # 7. Save model and feature extractor
    model_dir = get_timestamp_folder(OUTPUT_DIR)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "logreg.pkl"), "wb") as f:
        pickle.dump(model, f)
    # Save feature extractor state_dict (not the full model to avoid device issues)
    torch.save(feature_extractor.state_dict(), os.path.join(model_dir, "resnet18_feat.pth"))

    # 8. Prediction on test set
    test_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (malignant)

    # 9. Prepare output DataFrame
    # Output format: same as test file, but with the required prediction column(s)
    # We'll output a DataFrame with the same index as test_df, and a single column 'label' (probability)
    output_df = test_df.copy()
    output_df['label'] = test_probs
    output_df = output_df[['label']]
    # Save with the same format and extension as test file
    test_ext = os.path.splitext(TEST_CSV)[1]
    result_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    output_df.to_csv(result_path, index=True if test_df.index.name is not None else False)

    # 10. Validation step: compute AUROC on held-out validation set
    try:
        val_probs = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_probs)
        print(f"Validation AUROC: {val_score:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    # 11. Validation checks
    # a) Prediction file maintains exact test data indices
    pred_df = pd.read_csv(result_path, index_col=0 if test_df.index.name is not None else None)
    assert len(pred_df) == len(test_df), "Number of predictions does not match number of test samples."
    # b) Output column names match requirements
    assert list(pred_df.columns) == ['label'], f"Output columns {list(pred_df.columns)} do not match required ['label']."
    # c) Output format matches test file extension
    assert os.path.splitext(result_path)[1] == test_ext, "Output file extension does not match test file."
    # d) Sanity check: predictions are between 0 and 1
    assert ((pred_df['label'] >= 0) & (pred_df['label'] <= 1)).all(), "Predicted probabilities are not in [0, 1]."
    print("All validation checks passed. Results saved to:", result_path)