import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.label_bin, dtype=torch.float32)
        return img, label


def predict(image_paths, model, device, transform):
    """
    Predict malignant probability for a list of image file paths.
    Args:
      image_paths (list of str): paths to image files.
      model (torch.nn.Module): trained binary classifier outputting logits.
      device (torch.device): device to run inference on.
      transform (torchvision.transforms): image transforms.
    Returns:
      np.ndarray of shape (len(image_paths),) with probabilities in [0,1].
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            logits = model(img)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
    return np.array(probs)


def main():
    seed_everything(42)
    # Config
    INPUT_DIR = "./input"
    IMG_DIR = os.path.join(INPUT_DIR, "MyImages")
    CSV_PATH = os.path.join(INPUT_DIR, "mydataset.csv")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-4

    # Load data
    df = pd.read_csv(CSV_PATH)
    df["label_bin"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_bin"], random_state=42
    )

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Datasets and loaders
    train_ds = SkinLesionDataset(train_df, IMG_DIR, train_transform)
    val_ds = SkinLesionDataset(val_df, IMG_DIR, val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model: pretrained ResNet18
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(model.layer4.parameters()) + list(model.fc.parameters()), lr=LR
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")

    # Validation with Test-Time Augmentation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits1 = model(imgs).squeeze(1)
            imgs_flipped = torch.flip(imgs, dims=[3])
            logits2 = model(imgs_flipped).squeeze(1)
            probs1 = torch.sigmoid(logits1)
            probs2 = torch.sigmoid(logits2)
            avg_probs = ((probs1 + probs2) / 2).cpu().numpy()
            all_preds.extend(avg_probs.tolist())
            all_trues.extend(labels.numpy().tolist())

    auc = roc_auc_score(all_trues, all_preds)
    print(f"Validation AUROC with CosineAnnealingLR: {auc:.4f}")


if __name__ == "__main__":
    main()
