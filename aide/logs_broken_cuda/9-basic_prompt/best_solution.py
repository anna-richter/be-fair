import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.image_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = float(row.label_bin)
        return image, label


def predict(
    image_folder: str, model: nn.Module, device: torch.device, transform
) -> pd.DataFrame:
    """
    Predict malignancy probability for all .jpg images in image_folder using TTA (horizontal flip).
    Returns a DataFrame with columns ['image_name', 'probability'].
    """
    model.eval()
    results = []
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        image = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        img_t = transform(image).unsqueeze(0).to(device)  # 1,C,H,W
        with torch.no_grad():
            logits1 = model(img_t)
            logits2 = model(torch.flip(img_t, dims=[3]))
            avg_logit = (logits1 + logits2) / 2
            prob = torch.sigmoid(avg_logit).cpu().item()
        results.append({"image_name": os.path.splitext(fname)[0], "probability": prob})
    return pd.DataFrame(results)


def main():
    # Paths and device
    DATA_CSV = "./input/mydataset.csv"
    IMG_DIR = "./input/MyImages"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare data
    df = pd.read_csv(DATA_CSV)
    df["label_bin"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_bin"], random_state=42
    )

    # Transforms
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Datasets and loaders
    train_ds = SkinLesionDataset(train_df, IMG_DIR, transform=train_tf)
    val_ds = SkinLesionDataset(val_df, IMG_DIR, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model setup: EfficientNet-B0 backbone
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Validation with TTA
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits1 = model(imgs).squeeze(1)
            logits2 = model(torch.flip(imgs, dims=[3])).squeeze(1)
            avg_logits = (logits1 + logits2) / 2
            probs = torch.sigmoid(avg_logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation AUROC with TTA: {auc:.4f}")

    # If there's a test folder, save submission
    TEST_FOLDER = "./input/test_images"
    if os.path.isdir(TEST_FOLDER):
        sub = predict(TEST_FOLDER, model, DEVICE, val_tf)
        os.makedirs("./working", exist_ok=True)
        sub.to_csv("./working/submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()
