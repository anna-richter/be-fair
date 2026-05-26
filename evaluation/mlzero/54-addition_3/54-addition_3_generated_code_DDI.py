"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a deep learning image classifier (MobileNetV3) to predict the probability of malignancy for skin lesion images.
It uses PyTorch and timm for model training, and is optimized for multi-GPU and multi-CPU environments.
The script:
- Loads and preprocesses the data (removes NA labels, drops index columns, maps image paths)
- Extracts image features using a pretrained MobileNetV3 backbone
- Trains a classifier head on these features
- Saves the trained model to a timestamped folder in the specified output directory
- Predicts malignancy probabilities for the test set, preserving original indices and output format
- Saves predictions to the output directory, matching the test file format and column names
- Performs validation (AUROC) on a held-out validation set if training labels are available
- Includes validation checks to ensure output correctness

# Installation (uncomment if running in a fresh environment)
# !pip install torch torchvision timm pandas scikit-learn

"""

import os
import time
import uuid
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Paths
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/54-addition_3/node_15/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/54-addition_3"
# end change
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "MyImages")

# Use all available GPUs if possible
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Limit DataLoader workers to 2 to avoid resource contention on NFS
NUM_WORKERS = 2

def get_image_path(image_name):
    """Return absolute path to image file given image_name (without extension)."""
    return os.path.join(IMAGES_DIR, f"{image_name}.jpg")

def map_label_to_binary(label):
    """Map label string to binary: malignant=1, non-neoplastic=0."""
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == "malignant":
        return 1
    elif label == "non-neoplastic":
        return 0
    else:
        return np.nan

class SkinLesionDataset(Dataset):
    def __init__(self, df, image_col, label_col=None, transform=None):
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx][self.image_col]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.label_col is not None:
            label = self.df.iloc[idx][self.label_col]
            return image, label
        else:
            return image

def compute_class_weights(df, label_col="label_binary"):
    """Compute class weights for imbalanced data."""
    counts = df[label_col].value_counts().sort_index()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum()
    return torch.tensor([weights.get(0, 0.5), weights.get(1, 0.5)], dtype=torch.float32)

class MobileNetV3Classifier(nn.Module):
    def __init__(self, backbone_name="mobilenetv3_large_100", pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # Dynamically determine output feature size to avoid shape mismatch
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            in_features = features.shape[1] if len(features.shape) == 2 else features.shape[-1]
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

def train_model(model, train_loader, val_loader, class_weights, device, epochs=2, lr=1e-4):
    # Use pos_weight for BCEWithLogitsLoss: pos_weight should be a single float for binary classification
    # We use the ratio of negative to positive samples
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits)
                val_labels.append(labels.cpu().numpy())
                val_probs.append(probs.cpu().numpy())
        val_labels = np.concatenate(val_labels)
        val_probs = np.concatenate(val_probs)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            val_auc = 0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_auc

def predict_proba(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)

def main():
    # 1. Data Loading and Preprocessing
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Remove index columns if present
    for col in ['Unnamed: 0', 'index']:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    # Map image_name to absolute image path
    train['image'] = train['image_name'].apply(get_image_path)
    # start change
    # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
    test['image'] = test['image_name'].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.png")
    )
    # end change

    # Map label to binary (malignant=1, non-neoplastic=0)
    train['label_binary'] = train['label'].apply(map_label_to_binary)

    # Drop training samples without valid labels (do NOT drop from test)
    train = train[train['label_binary'].isin([0, 1])].reset_index(drop=True)

    # 2. Prepare training/validation split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(train, train['label_binary']))
    train_data = train.iloc[train_idx].reset_index(drop=True)
    val_data = train.iloc[val_idx].reset_index(drop=True)

    # 3. Compute class weights for fairness
    class_weights = compute_class_weights(train_data, label_col="label_binary").to(DEVICE)

    # 4. Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Datasets and Loaders
    batch_size = 32  # Reduce batch size to avoid OOM and speed up per-epoch time
    train_dataset = SkinLesionDataset(train_data, image_col="image", label_col="label_binary", transform=train_transform)
    val_dataset = SkinLesionDataset(val_data, image_col="image", label_col="label_binary", transform=val_transform)
    test_dataset = SkinLesionDataset(test, image_col="image", label_col=None, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 6. Model Training
    import timm
    timestamp = int(time.time() * 1000)
    model_dir = os.path.join(OUTPUT_DIR, f"torch_model_{timestamp}_{uuid.uuid4().hex[:8]}")
    os.makedirs(model_dir, exist_ok=True)

    # Use a smaller model to reduce training time and memory
    model = MobileNetV3Classifier(backbone_name="mobilenetv3_large_100", pretrained=True)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # Reduce epochs to 2 to avoid wall-clock timeout
    model, best_val_auc = train_model(model, train_loader, val_loader, class_weights, DEVICE, epochs=2, lr=1e-4)

    # Save model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # 7. Prediction
    # Predict probabilities for test set (malignancy probability)
    malignancy_prob = predict_proba(model, test_loader, DEVICE)

    # start change
    _ddi_df = pd.DataFrame({
        "DDI_file": test["image_name"].astype(str) + ".png",
        "predicted_probability": malignancy_prob.astype(float),
    }).reset_index(drop=True)
    _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
    # end change

    # Prepare output DataFrame
    results = test[['image_name']].copy()
    results['label'] = malignancy_prob

    # Save results in same format/extension as test.csv (CSV)
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    results.to_csv(results_path, index=False)

    # 8. Validation Checks
    # a) Check that output has same number of rows as test set
    assert len(results) == len(test), f"Prediction rows ({len(results)}) != test rows ({len(test)})"
    # b) Check that output indices match test set
    assert all(results['image_name'].values == test['image_name'].values), "Output image_name order does not match test set"
    # c) Check that columns match requirements
    assert list(results.columns) == ['image_name', 'label'], f"Output columns {list(results.columns)} do not match ['image_name', 'label']"
    # d) Check output format (CSV)
    assert results_path.endswith('.csv'), "Output file must be CSV"
    # e) Check that all probabilities are in [0, 1]
    assert np.all((results['label'] >= 0) & (results['label'] <= 1)), "Predicted probabilities not in [0, 1]"

    # 9. Validation Metric (AUROC) on held-out validation set
    try:
        val_dataset_pred = SkinLesionDataset(val_data, image_col="image", label_col=None, transform=val_transform)
        val_loader_pred = DataLoader(val_dataset_pred, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        val_probs = predict_proba(model, val_loader_pred, DEVICE)
        val_true = val_data['label_binary'].values
        val_auroc = roc_auc_score(val_true, val_probs)
        print(f"Validation AUROC: {val_auroc:.4f}")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")

if __name__ == "__main__":
    main()