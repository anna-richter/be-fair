import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row.image_name}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.float32)
        return img, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def predict(folder_path, model, device, mean, std):
    """
    Predict malignancy probability for all .jpg images in folder_path using TenCrop TTA.
    Returns a DataFrame and writes submission.csv to ./working.
    """
    model.eval()
    results = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
    resize = transforms.Resize(256)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    tencrop = transforms.TenCrop(224)
    with torch.no_grad():
        for fname in files:
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            crops = tencrop(resize(img))
            batch = torch.stack([normalize(to_tensor(c)) for c in crops]).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).mean().item()
            name = os.path.splitext(fname)[0]
            results.append((name, probs))
    sub_df = pd.DataFrame(results, columns=["image_name", "malignancy_prob"])
    os.makedirs("working", exist_ok=True)
    sub_df.to_csv("working/submission.csv", index=False)
    return sub_df


def main():
    # Settings
    DATA_CSV = "input/mydataset.csv"
    IMG_DIR = "input/MyImages"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Load data
    df = pd.read_csv(DATA_CSV)
    df["label"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # DataLoaders
    train_ds = SkinLesionDataset(train_df, IMG_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # Model: EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    in_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_ftrs, 1))
    model = model.to(DEVICE)

    # Loss & optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {epoch_loss:.4f}")

    # Validation with TenCrop TTA
    model.eval()
    resize = transforms.Resize(256)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    tencrop = transforms.TenCrop(224)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img = Image.open(os.path.join(IMG_DIR, f"{row.image_name}.jpg")).convert(
                "RGB"
            )
            crops = tencrop(resize(img))
            batch = torch.stack([normalize(to_tensor(c)) for c in crops]).to(DEVICE)
            logits = model(batch)
            prob = torch.sigmoid(logits).mean().item()
            all_preds.append(prob)
            all_labels.append(row.label)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Validation AUROC (TenCrop TTA, AdamW+WD): {auc:.4f}")

    # Final test-time-augmented prediction
    predict(IMG_DIR, model, DEVICE, mean, std)


if __name__ == "__main__":
    main()
