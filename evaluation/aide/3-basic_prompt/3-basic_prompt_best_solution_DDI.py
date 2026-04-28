import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform):
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
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


def predict(folder_path):
    """
    Predict malignancy probabilities for all JPEG images in folder_path using TTA.
    Returns a dict image_name -> probability (0–1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model.load_state_dict(torch.load("working/model.pt", map_location=device))
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    results = {}
    for fname in os.listdir(folder_path):
        # start change
        # if not fname.lower().endswith(".jpg"):  # original
        if not fname.lower().endswith((".jpg", ".png")):
        # end change
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        x_flip = torch.flip(x, dims=[3])
        with torch.no_grad():
            p1 = torch.sigmoid(model(x))
            p2 = torch.sigmoid(model(x_flip))
            results[fname] = ((p1 + p2) / 2.0).item()
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start change
    # df = pd.read_csv("input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df = df[df["label"].isin(["malignant", "benign"])].copy()
    df["label"] = df["label"].map({"benign": 0, "malignant": 1})
    # start change
    # img_dir = "input/MyImages"  # original
    img_dir = "/home/anri21/be-fair/aide/MyData/MyImages"
    # end change
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    epochs = 5
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_loader = DataLoader(
            SkinDataset(train_df, img_dir, transform),
            batch_size=32,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            SkinDataset(val_df, img_dir, transform),
            batch_size=32,
            shuffle=False,
            num_workers=2,
        )
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 1)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        model.train()
        for epoch in range(epochs):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                imgs_flip = torch.flip(imgs, dims=[3])
                logits1 = model(imgs)
                logits2 = model(imgs_flip)
                probs = torch.sigmoid((logits1 + logits2) / 2.0).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_targets.extend(labels.numpy().flatten())
        auc = roc_auc_score(all_targets, all_probs)
        print(f"Fold {fold+1} ROC AUC: {auc:.4f}")
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    print(f"Mean CV ROC AUC: {mean_auc:.4f}")

    full_loader = DataLoader(
        SkinDataset(df, img_dir, transform), batch_size=32, shuffle=True, num_workers=2
    )
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(full_loader)
    )
    model.train()
    for epoch in range(epochs):
        for imgs, labels in full_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    os.makedirs("working", exist_ok=True)
    torch.save(model.state_dict(), "working/model.pt")


if __name__ == "__main__":
    main()
    # start change
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [float(_preds[f]) for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
