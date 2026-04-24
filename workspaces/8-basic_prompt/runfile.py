import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Dataset for skin lesion images
class SkinDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row.image_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row.label, dtype=torch.float32)
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate_with_tta(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            # original predictions
            logits1 = model(imgs)
            # horizontal flip predictions
            logits2 = model(torch.flip(imgs, dims=[3]))
            probs = (torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0
            probs = probs.cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.numpy().flatten().tolist())
    auc = roc_auc_score(all_targets, all_probs)
    return auc


def predict(image_paths, model_path="working/model.pth", batch_size=32):
    """
    Given a list of image file paths, load the trained model and return
    a dict {image_path: malignancy_probability} using TTA (horizontal flip).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    results = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits1 = model(tensor)
            logits2 = model(torch.flip(tensor, dims=[3]))
            probs = (
                ((torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2.0)
                .cpu()
                .numpy()
                .flatten()
            )
        for p, prob in zip(batch_paths, probs):
            results[p] = prob
    return results


def main():
    # Paths and device
    csv_path = "./input/mydataset.csv"
    image_dir = "./input/MyImages"
    os.makedirs("working", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].map(lambda x: 1 if x == "malignant" else 0)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Transforms
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets and loaders
    train_ds = SkinDataset(train_df, image_dir, transform=train_tfms)
    val_ds = SkinDataset(val_df, image_dir, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop with TTA evaluation
    best_auc = 0.0
    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = evaluate_with_tta(model, val_loader, device)
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val AUROC (TTA)={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "working/model.pth")

    print(f"Best Validation AUROC (with TTA): {best_auc:.4f}")


if __name__ == "__main__":
    main()
