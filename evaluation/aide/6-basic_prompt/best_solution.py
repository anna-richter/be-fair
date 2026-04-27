import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["target"], dtype=torch.float32)
        return image, label


def predict(folder_path: str, model_path: str = "working/model.pth"):
    """
    Load the trained model and predict malignancy probability for all images in folder_path
    using 10-crop TTA (5 crops + their horizontal flips).
    Saves a CSV to working/submission.csv with columns: image_name, malignancy_probability.
    Returns a pandas DataFrame of the predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    def ten_crop_transform(img):
        crops = transforms.Resize(256)(img)
        crops = transforms.TenCrop(224)(crops)
        batch = []
        for crop in crops:
            t = transforms.ToTensor()(crop)
            t = normalize(t)
            batch.append(t)
        return torch.stack(batch)

    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    results = []
    with torch.no_grad():
        for fname in tqdm(files, desc="Predicting with 10-Crop TTA"):
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            crops = ten_crop_transform(img).to(device)
            logits = model(crops)
            probs = torch.sigmoid(logits).mean().item()
            name = os.path.splitext(fname)[0]
            results.append({"image_name": name, "malignancy_probability": probs})
    df_pred = pd.DataFrame(results)
    os.makedirs("working", exist_ok=True)
    df_pred.to_csv("working/submission.csv", index=False)
    return df_pred


def main():
    # Paths
    csv_path = "input/mydataset.csv"
    img_dir = "input/MyImages"

    # Load labels
    df = pd.read_csv(csv_path)
    df["target"] = (df["label"] == "malignant").astype(int)

    # Stratified hold-out
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, df["target"]))
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    # Compute sample weights for WeightedRandomSampler
    class_counts = df_train["target"].value_counts().to_dict()
    weights = df_train["target"].apply(lambda x: 1.0 / class_counts[x]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )
    normalize = transforms.Normalize(mean, std)

    def preprocess_ten_crop(crops):
        processed = []
        for crop in crops:
            t = transforms.ToTensor()(crop)
            t = normalize(t)
            processed.append(t)
        return torch.stack(processed)

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(preprocess_ten_crop),
        ]
    )

    # Datasets & Loaders
    train_ds = SkinDataset(df_train, img_dir, transform=train_transform)
    val_ds = SkinDataset(df_val, img_dir, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model, loss, optimizer, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"
        ):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Validation with 10-Crop TTA
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating with 10-Crop TTA"):
            bs = images.size(0)
            images = images.view(-1, 3, 224, 224).to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).view(bs, 10)
            avg = probs.mean(dim=1)
            all_probs.extend(avg.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation AUROC with WeightedSampler + 10-Crop TTA: {roc_auc:.4f}")

    # Save model
    os.makedirs("working", exist_ok=True)
    torch.save(model.state_dict(), "working/model.pth")


if __name__ == "__main__":
    main()
