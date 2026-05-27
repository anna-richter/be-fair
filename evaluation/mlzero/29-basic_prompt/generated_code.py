"""
Skin Lesion Malignancy Probability Prediction Script

This script trains a deep learning image classifier (EfficientNet) using PyTorch and timm to predict the probability of skin lesion malignancy from RGB JPEG images. It:
- Loads and preprocesses the training data (removes NA labels, drops unnecessary index columns)
- Maps string labels to binary (malignant=1, others=0)
- Trains a binary classifier on images
- Saves the trained model to a timestamped folder in the specified output directory
- Predicts malignancy probabilities for the test set, preserving original indices and output format
- Saves predictions to a file named "results" with the same extension as the test file in the output directory
- Performs validation checks to ensure output integrity
- If training data is available, holds out 10% for validation and prints AUROC on the validation set

Installation requirements (run in bash before running this script):
    pip install torch torchvision timm pandas scikit-learn tqdm pillow

Usage:
    Place this script in an environment with access to the data and run as __main__.
"""

import os
import time
import random
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Constants
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/29-basic_prompt/node_13/output"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "MyImages")

# Use all CPUs for dataloader, 1 GPU for training
NUM_WORKERS = min(8, os.cpu_count())  # Reduce workers for NFS stability and wall-clock
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_path(image_name):
    """Returns absolute path to image file given image_name (without extension)."""
    return os.path.join(IMAGES_DIR, f"{image_name}.jpg")

def map_label_to_binary(label):
    """Maps string label to binary: malignant=1, others=0."""
    return 1 if str(label).strip().lower() == "malignant" else 0

class SkinLesionDataset(Dataset):
    """Custom Dataset for skin lesion images."""
    def __init__(self, df, img_col='image', label_col=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, self.img_col]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.label_col is not None:
            label = self.df.loc[idx, self.label_col]
            return image, label
        else:
            return image

def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed_everything(42)

    # 1. Data Loading and Preprocessing
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Remove unnecessary index column if present
    for df in [train_df, test_df]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Drop NA labels from training data only
    train_df = train_df.dropna(subset=['label'])

    # Map string labels to binary (malignant=1, others=0)
    train_df['label'] = train_df['label'].apply(map_label_to_binary)

    # Add absolute image paths for training and test data
    train_df['image'] = train_df['image_name'].apply(get_image_path)
    test_df['image'] = test_df['image_name'].apply(get_image_path)

    # Only keep columns needed for training
    train_data = train_df[['image', 'label']].copy()
    test_data = test_df[['image_name', 'image']].copy()
    test_data_indices = test_df.index.copy()

    # 2. Validation Split (if no separate validation set is provided)
    if len(train_data) > 10:
        train_data, val_data = train_test_split(
            train_data, test_size=0.1, random_state=42, stratify=train_data['label']
        )
        has_validation = True
    else:
        val_data = None
        has_validation = False

    # 3. Model Training
    # Use a random timestamp for model folder
    timestamp = int(time.time()) + random.randint(0, 9999)
    model_dir = os.path.join(OUTPUT_DIR, f"skin_model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")

    # Image transforms
    img_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets and loaders
    train_dataset = SkinLesionDataset(train_data, img_col='image', label_col='label', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    if has_validation:
        val_dataset = SkinLesionDataset(val_data, img_col='image', label_col='label', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model: EfficientNet-B0 (fast, accurate, small)
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    # FIX: Remove verbose=True (not supported in some torch versions)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    # Training loop (limit epochs to fit within 1 hour wall-clock)
    max_epochs = 4  # Reduce epochs for wall-clock safety
    best_val_auc = 0
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().to(DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # Validation
        if has_validation:
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    val_preds.extend(probs)
                    val_targets.extend(labels.numpy())
            try:
                val_auc = roc_auc_score(val_targets, val_preds)
                print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val AUROC={val_auc:.4f}")
                scheduler.step(val_auc)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), model_path)
            except Exception as e:
                print(f"Validation failed: {e}")
        else:
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}")

    # Save final model if not already saved
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)

    # 4. Prediction on Test Set
    # Use best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Test dataset and loader
    test_dataset = SkinLesionDataset(test_data, img_col='image', label_col=None, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    test_probs = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting on test set"):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            test_probs.extend(probs)

    # Prepare output DataFrame
    results_df = pd.DataFrame({
        "label": test_probs
    }, index=test_data_indices)

    # 5. Save Results
    test_ext = os.path.splitext(TEST_CSV)[1]
    results_path = os.path.join(OUTPUT_DIR, f"results{test_ext}")
    results_df.to_csv(results_path, index=True, header=True)

    # 6. Validation Checks
    loaded_results = pd.read_csv(results_path, index_col=0)
    assert len(loaded_results) == len(test_df), \
        f"Number of predictions ({len(loaded_results)}) does not match test set ({len(test_df)})"
    assert list(loaded_results.index) == list(test_df.index), \
        "Prediction file indices do not match test set indices"
    assert list(loaded_results.columns) == ["label"], \
        f"Prediction file columns {list(loaded_results.columns)} do not match required ['label']"
    assert results_path.endswith('.csv'), "Output file format does not match test file format"
    assert np.all((loaded_results['label'] >= 0) & (loaded_results['label'] <= 1)), \
        "Predicted probabilities are not all in [0, 1]"

    # 7. Validation Metric (if applicable)
    if has_validation:
        try:
            val_dataset = SkinLesionDataset(val_data, img_col='image', label_col='label', transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    val_preds.extend(probs)
                    val_targets.extend(labels.numpy())
            val_auc = roc_auc_score(val_targets, val_preds)
            print(f"Final Validation AUROC: {val_auc:.4f}")
        except Exception as e:
            print(f"Validation failed: {e}")

    print(f"Predictions saved to: {results_path}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()